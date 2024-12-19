import argparse
import os
import random
import signal
import sys
from typing import Dict, List, Optional, Tuple
sys.path.append('/data/workspace/TX-DNN-Solution-main/DeepFilterNet/')
import numpy as np
import torch
import torchaudio
from loguru import logger
from torch import Tensor, nn, optim
from torch.autograd.anomaly_mode import set_detect_anomaly
from torch.autograd.grad_mode import set_grad_enabled
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.types import Number

from df.checkpoint import check_patience, load_model, read_cp, write_cp
from df.config import Csv, config
from df.logger import init_logger, log_metrics, log_model_summary
from df.loss import Istft, Loss, Stft
from df.lr import cosine_scheduler
from df.model import ModelParams
from df.modules import get_device
from df.utils import (
    as_complex,
    as_real,
    check_finite_module,
    check_manual_seed,
    detach_hidden,
    get_host,
    get_norm_alpha,
    make_np,
)
from libdf import DF
from libdfdata import PytorchDataLoader as DataLoader
from dataloader import load_data_with_list
from df.erb_py import PY_DF
from utils_signal_proc import MDCT_signal,IMDCT_signal
from soundfile import read, write

from frequencydis import MultiFrequencyDiscriminator
from df.loss import Disc_Loss,Gen_FM_Loss, DNSMOS_Loss

model_Disc = MultiFrequencyDiscriminator(1, [128, 256, 512, 1024]).cuda()


should_stop = False
debug = False
log_timings = False
state: Optional[DF] = None
istft: Optional[nn.Module]
MAX_NANS = 50

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3, 4, 5, 6, 7"
torch.backends.cudnn.enabled = False



@logger.catch
def main():
    global should_stop, debug, state, log_timings

    parser = argparse.ArgumentParser()
    parser.add_argument("data_config_file", type=str, help="Path to a dataset config file.")
    parser.add_argument(
        "data_dir", type=str, help="Path to the dataset directory containing .hdf5 files."
    )
    parser.add_argument(
        "base_dir", type=str, help="Directory to store logs, summaries, checkpoints, etc."
    )
    parser.add_argument(
        "--train_speech_list", type=str, help="which contain the file list for training speech"
    )
    parser.add_argument(
        "--train_noise_list", type=str, help="which contain the file list for training noise"
    )
    parser.add_argument(
        "--host-batchsize-config",
        "-b",
        type=str,
        default=None,
        help="Path to a host specific batch size config.",
    )
    parser.add_argument("--no-resume", action="store_false", dest="resume")
    parser.add_argument(
        "--log-level",
        type=str,
        default=None,
        help="Logger verbosity. Can be one of (trace, debug, info, error, none)",
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--no-debug", action="store_false", dest="debug")
    args = parser.parse_args()
    if not os.path.isfile(args.data_config_file):
        raise FileNotFoundError("Dataset config not found at {}".format(args.data_config_file))
    if not os.path.isdir(args.data_dir):
        NotADirectoryError("Data directory not found at {}".format(args.data_dir))
    os.makedirs(args.base_dir, exist_ok=True)
    summary_dir = os.path.join(args.base_dir, "summaries")
    os.makedirs(summary_dir, exist_ok=True)
    debug = args.debug
    if args.log_level is not None:
        if debug and args.log_level.lower() != "debug":
            raise ValueError("Either specify debug or a manual log level")
        log_level = args.log_level
    else:
        log_level = "DEBUG" if debug else "INFO"
    init_logger(file=os.path.join(args.base_dir, "train.log"), level=log_level, model=args.base_dir)
    config_file = os.path.join(args.base_dir, "config.ini")
    config.load(config_file)
    seed = config("SEED", 42, int, section="train")
    check_manual_seed(seed)
    logger.info("Running on device {}".format(get_device()))

    # Maybe update batch size
    if args.host_batchsize_config is not None:
        try:
            sys.path.append(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            )
            from scripts.set_batch_size import main as set_batch_size  # type: ignore

            key = get_host() + "_" + config.get("model", section="train")
            key += "_" + config.get("fft_size", section="df")
            set_batch_size(config_file, args.host_batchsize_config, host_key=key)
            config.load(config_file, allow_reload=True)  # Load again
        except Exception as e:
            logger.error(f"Could not apply host specific batch size config: {str(e)}")

    signal.signal(signal.SIGUSR1, get_sigusr1_handler(args.base_dir))

    p = ModelParams()
    state = DF(
        sr=p.sr,
        fft_size=p.fft_size,
        hop_size=p.hop_size,
        nb_bands=p.nb_erb,
        min_nb_erb_freqs=p.min_nb_freqs,
    )
    checkpoint_dir = os.path.join(args.base_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    mask_only: bool = config("MASK_ONLY", False, bool, section="train")
    train_df_only: bool = config("DF_ONLY", False, bool, section="train")
    jit = config("JIT", False, cast=bool, section="train")
    model, epoch = load_model(
        checkpoint_dir if args.resume else None,
        state,
        jit=False,
        mask_only=mask_only,
        train_df_only=train_df_only,
    )

    bs: int = config("BATCH_SIZE", "batch_size", int, section="train")
    bs_eval: int = config("BATCH_SIZE_EVAL", "batch_size_eval", int, section="train")
    bs_eval = bs_eval if bs_eval > 0 else bs
    overfit = config("OVERFIT", False, bool, section="train")
    log_timings = config("LOG_TIMINGS", False, bool, section="train", save=False)

    tr_dataloader = load_data_with_list(args.data_dir,
                                        args.train_speech_list,
                                        args.train_noise_list,
                                        bs, config("NUM_WORKERS", 4, int, section="train"), 3 * p.sr, p.sr)

    val_dataloader = load_data_with_list(args.data_dir,
                                         '/data/workspace/TX-DNN-Solution-main/Dataset/cv_speech_32k.list',
                                         '/data/workspace/TX-DNN-Solution-main/Dataset/cv_noise_32k.list',
                                         bs, config("NUM_WORKERS", 4, int, section="train"), 3 * p.sr, p.sr)

    ts_dataloader = load_data_with_list(args.data_dir,
                                        '/data/workspace/TX-DNN-Solution-main/Dataset/tt_speech_32k.list',
                                        '/data/workspace/TX-DNN-Solution-main/Dataset/tt_noise_32k.list',
                                        bs, config("NUM_WORKERS", 4, int, section="train"), 3 * p.sr, p.sr)

    # Batch size scheduling limits the batch size for the first epochs. It will increase the batch
    # size during training as specified. Used format is a comma separated list containing
    # epoch/batch size tuples where each tuple is separated via '/':
    # '<epoch>/<batch_size>,<epoch>/<batch_size>,<epoch>/<batch_size>'
    # The first epoch has to be 0, later epoch may modify the batch size as specified.
    # This only applies to training batch size.
    batch_size_scheduling: List[str] = config("BATCH_SIZE_SCHEDULING", [], Csv(str), section="train")  # type: ignore
    scheduling_bs = bs
    prev_scheduling_bs = bs
    if len(batch_size_scheduling) > 0:
        batch_size_scheduling = [
            (int(bs[0]), int(bs[1])) for bs in (bs.split("/") for bs in batch_size_scheduling)
        ]
        assert batch_size_scheduling[0][0] == 0  # First epoch must be 0
        logger.info("Running with learning rate scheduling")

    max_epochs = config("MAX_EPOCHS", 10, int, section="train")
    assert epoch >= 0
    opt = load_opt(
        checkpoint_dir if args.resume else None,
        model,
        mask_only,
        train_df_only,
    )
    lrs = setup_lrs(len(tr_dataloader))
    wds = setup_wds(len(tr_dataloader))
    if not args.resume and os.path.isfile(os.path.join(checkpoint_dir, ".patience")):
        os.remove(os.path.join(checkpoint_dir, ".patience"))
    try:
        log_model_summary(model, verbose=args.debug)
    except Exception as e:
        logger.warning(f"Failed to print model summary: {e}")
    if jit:
        # Load as jit after log_model_summary
        model = torch.jit.script(model)

    # Validation optimization target. Used for early stopping and selecting best checkpoint
    val_criteria = []
    val_criteria_type = config("VALIDATION_CRITERIA", "loss", section="train")  # must be in metrics
    val_criteria_rule = config("VALIDATION_CRITERIA_RULE", "min", section="train")
    val_criteria_rule = val_criteria_rule.replace("less", "min").replace("more", "max")
    patience = config("EARLY_STOPPING_PATIENCE", 5, int, section="train")

    losses = setup_losses()
    
    if config("START_EVAL", False, cast=bool, section="train"):
        val_loss = run_epoch(
            model=model,
            epoch=epoch - 1,
            loader=val_dataloader,
            split="valid",
            opt=opt,
            losses=losses,
            summary_dir=summary_dir,
        )
        metrics = {"loss": val_loss}
        metrics.update(
            {n: torch.mean(torch.stack(vals)).item() for n, vals in losses.get_summaries()}
        )
        log_metrics(f"[{epoch - 1}] [valid]", metrics)
    losses.reset_summaries()
    # Save default values to disk
    config.save(os.path.join(args.base_dir, "config.ini"))
    
    for epoch in range(epoch, max_epochs):
        if len(batch_size_scheduling) > 0:
            # Get current batch size
            for e, b in batch_size_scheduling:
                if e <= epoch:
                    # Update bs, but don't go higher than the batch size specified in the config
                    scheduling_bs = min(b, bs)
            if prev_scheduling_bs != scheduling_bs:
                logger.info(f"Batch scheduling | Setting batch size to {scheduling_bs}")
                tr_dataloader.set_batch_size(scheduling_bs, "train")
                # Update lr/wd scheduling since dataloader len changed
                lrs = setup_lrs(len(tr_dataloader))
                wds = setup_wds(len(tr_dataloader))
                prev_scheduling_bs = scheduling_bs
        train_loss = run_epoch(
            model=model,
            epoch=epoch,
            loader=tr_dataloader,
            split="train",
            opt=opt,
            losses=losses,
            summary_dir=summary_dir,
            lr_scheduler_values=lrs,
            wd_scheduler_values=wds,
        )
        metrics = {"loss": train_loss}
        try:
            metrics["lr"] = opt.param_groups[0]["lr"]
        except AttributeError:
            pass
        if debug:
            metrics.update(
                {n: torch.mean(torch.stack(vals)).item() for n, vals in losses.get_summaries()}
            )
        log_metrics(f"[{epoch}] [train]", metrics)
        write_cp(model, "model", checkpoint_dir, epoch + 1)
        write_cp(opt, "opt", checkpoint_dir, epoch + 1)
        losses.reset_summaries()
        val_loss = run_epoch(
            model=model,
            epoch=epoch,
            loader=val_dataloader,
            split="valid",
            opt=opt,
            losses=losses,
            summary_dir=summary_dir,
        )
        metrics = {"loss": val_loss}
        metrics.update(
            {n: torch.mean(torch.stack(vals)).item() for n, vals in losses.get_summaries()}
        )
        val_criteria = metrics[val_criteria_type]
        write_cp(
            model, "model", checkpoint_dir, epoch + 1, metric=val_criteria, cmp=val_criteria_rule
        )
        log_metrics(f"[{epoch}] [valid]", metrics)
        if not check_patience(
            checkpoint_dir,
            max_patience=patience,
            new_metric=val_criteria,
            cmp=val_criteria_rule,
            raise_=False,
        ):
            break
        if should_stop:
            logger.info("Stopping training due to timeout")
            exit(0)
        losses.reset_summaries()
    model, epoch = load_model(
        checkpoint_dir,
        state,
        jit=jit,
        epoch="best",
        mask_only=mask_only,
        train_df_only=train_df_only,
    )
    test_loss = run_epoch(
        model=model,
        epoch=epoch,
        loader=ts_dataloader,
        split="test",
        opt=opt,
        losses=losses,
        summary_dir=summary_dir,
    )
    metrics: Dict[str, Number] = {"loss": test_loss}
    metrics.update({n: torch.mean(torch.stack(vals)).item() for n, vals in losses.get_summaries()})
    log_metrics(f"[{epoch}] [test]", metrics)
    logger.info("Finished training")


def run_epoch(
    model: nn.Module,
    epoch: int,
    loader: DataLoader,
    split: str,
    opt: optim.Optimizer,
    losses: Loss,
    summary_dir: str,
    lr_scheduler_values: Optional[np.ndarray] = None,
    wd_scheduler_values: Optional[np.ndarray] = None,
) -> float:
    global debug
    
    log_freq = config("LOG_FREQ", cast=int, default=100, section="train")
    logger.info("Start epoch {}".format(epoch))
    detect_anomaly: bool = config("DETECT_ANOMALY", False, bool, section="train")
    if detect_anomaly:
        logger.info("Running with autograd profiling")
        
    l_mem = []
    Gl_mem = []
    p = ModelParams()

    dev = get_device()
    
    py_df = PY_DF(sr=p.sr, fft_size=p.fft_size, hop_size=p.hop_size, nb_erb=p.nb_erb,
                       nb_df=p.nb_df, min_nb_freqs=p.min_nb_freqs, device=dev)

    is_train = split == "train"
    
    opt_Disc = optim.AdamW(model_Disc.parameters(), lr=5e-4, weight_decay=0.05, betas=[0.9, 0.999], amsgrad=True)
    loss_Disc = Disc_Loss()
    loss_Gen = Gen_FM_Loss()
    #loss_MOS = DNSMOS_Loss()
    model.train(mode=is_train)
    losses.store_losses = debug or not is_train
    start_steps = epoch
    max_steps = len(loader) - 1

    for i, data in enumerate(loader):
        opt.zero_grad()
        opt_Disc.zero_grad()
        it = start_steps + i  # global training iteration
        if lr_scheduler_values is not None or wd_scheduler_values is not None:
            for param_group in opt.param_groups:
                if lr_scheduler_values is not None:
                    param_group["lr"] = lr_scheduler_values[it] * param_group.get("lr_scale", 1)
                if wd_scheduler_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_scheduler_values[it]

        target = data[0].to(dev)
        noisy = data[1].to(dev)
        #shape torch.Size([128, 151, 1026])
        noisy_spec = MDCT_signal(noisy, p.fft_size, p.hop_size, use_power_compress=False)
        target_spec = MDCT_signal(target, p.fft_size, p.hop_size, use_power_compress=False) 
        noisy_rev = IMDCT_signal(noisy_spec, p.fft_size, p.hop_size, use_power_compress=False)
        #target_rev = IMDCT_signal(target_spec, p.fft_size, p.hop_size, use_power_compress=False)
        

               
        b, t, f = noisy_spec.shape
        noisy_spec_complex = torch.view_as_complex(noisy_spec.view(b, t, -1, 2))
        noisy_spec_complex = noisy_spec_complex * (1 / p.fft_size)
        target_spec_complex = torch.view_as_complex(target_spec.view(b, t, -1, 2)) #shape torch.Size([128, 151, 513, 2])
        target_spec_complex = target_spec_complex * (1 / p.fft_size)

        
            
        # py version
        noisy_power = torch.as_tensor(noisy_spec_complex)[:, :, :].abs().square()
        noisy_erb = torch.matmul(noisy_power, py_df.fb)
        noisy_erb_feat = py_df.unit_norm_erb_pydf(noisy_erb)
        noisy_erb_feat = torch.as_tensor(noisy_erb_feat).unsqueeze(1)
        noisy_spec_feat = py_df.unit_norm_pydf(
            torch.view_as_real(torch.as_tensor(noisy_spec_complex[:, :, :p.nb_df])))

        noisy_spec_feat = as_real(torch.as_tensor(noisy_spec_feat).unsqueeze(1))
        noisy_spec = as_real(torch.as_tensor(noisy_spec_complex).unsqueeze(1))

        with set_detect_anomaly(detect_anomaly and is_train), set_grad_enabled(is_train):
            if not is_train:
                input = noisy_spec.clone()
            else:
                input = noisy_spec

            enh, m, lsnr, other = model.forward(
                spec=input,
                feat_erb=noisy_erb_feat,
                feat_spec=noisy_spec_feat,
            )
            
            snrs = data[-1].to(dev).clone().detach()
            noisy_spec_complex = noisy_spec_complex.unsqueeze(1)
            target_spec_complex = target_spec_complex.unsqueeze(1)
            #print("enh - target",enh.shape,target.shape)
            est_wav = IMDCT_signal(enh / (1 / p.fft_size), p.fft_size, p.hop_size, use_power_compress=False).squeeze(1)
            #val_dir = './wav_check/'
            #write(os.path.join(val_dir, str(i) + "iter_5_target.wav"), target[5].cpu().numpy(), 32000)
            #write(os.path.join(val_dir, str(i) + "iter_5_est.wav"), est_wav[5].detach().cpu().numpy(), 32000)
            target_out, _ = model_Disc(target, sample_rate=p.sr)
            est_out, _ = model_Disc(est_wav.detach(), sample_rate=p.sr)
            Loss_D = loss_Disc(target_out, est_out, target, est_wav.detach())
            
            if is_train:
                try:
                    Loss_D.backward()
                    clip_grad_norm_(model_Disc.parameters(), 1.0, error_if_nonfinite=True)
                except:
                    print("Disc loss wrong")
            opt_Disc.step()
            # finetune update Generator weights
            err1 = torch.tensor((0),dtype=torch.float32).to(enh.device)
            if is_train:
                target_outs, target_feat_map = model_Disc(target, sample_rate=p.sr)
                est_outs, est_feat_map = model_Disc(est_wav, sample_rate=p.sr)
                err1 = loss_Gen(est_outs, target_outs, est_feat_map, target_feat_map)
                #err2 = loss_MOS(est_wav.detach(),target)
                #print(err2)

            try:
                err = losses.forward(target_spec_complex, noisy_spec_complex, enh, m, lsnr, snrs=snrs)
                err += err1
            except Exception as e:
                if "nan" in str(e).lower() or "finite" in str(e).lower():
                    logger.warning("NaN in loss computation: {}. Skipping backward.".format(str(e)))
                    check_finite_module(model)
                    n_nans += 1
                    if n_nans > MAX_NANS:
                        raise e
                    continue
                raise e

            if is_train:
                try:
                    err.backward() #retain_graph=True
                    clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                except RuntimeError as e:
                    e_str = str(e)
                    if "nan" in e_str.lower() or "non-finite" in e_str:
                        check_finite_module(model)
                        logger.error(e_str)
                        os.makedirs(os.path.join(summary_dir, "nan"), exist_ok=True)
                        for batch_idx in range(clean.shape[0]):
                            clean_idx = batch.ids[batch_idx].item()
                            summary_write(
                                target_spec_complex.detach(),
                                noisy_spec_complex.detach(),
                                enh.detach(),
                                batch.snr.detach(),
                                lsnr.detach().float(),
                                os.path.join(summary_dir, "nan"),
                                prefix=split + f"_e{epoch}_i{i}_b{batch_idx}_ds{clean_idx}",
                                idx=batch_idx,
                            )
                        cleanup(err, noisy_spec_complex, target_spec_complex, enh, m, feat_erb, feat_spec, batch)
                        n_nans += 1
                        if n_nans > MAX_NANS:
                            raise e
                        continue
                    else:
                        raise e
                opt.step()
            detach_hidden(model)

        l_mem.append(err.detach())
        Gl_mem.append(err1.detach())
        if i % log_freq == 0:
            l_mean = torch.stack(l_mem[-100:]).mean().cpu()
            Gl_mean = torch.stack(Gl_mem[-100:]).mean().cpu()
            if torch.isnan(l_mean):
                check_finite_module(model)
            l_dict = {"loss": l_mean.item()}
            if err1 is not None:
                l_dict["G_loss"] = Gl_mean.item()
            if lr_scheduler_values is not None:
                l_dict["lr"] = opt.param_groups[0]["lr"]
            if wd_scheduler_values is not None:
                l_dict["wd"] = opt.param_groups[0]["weight_decay"]
            if log_timings:
                l_dict["t_sample"] = batch.timings[:-1].sum()
                l_dict["t_batch"] = batch.timings[-1].mean()  # last is for whole batch
            if debug:
                l_dict.update(
                    {
                        n: torch.mean(torch.stack(vals[-bs:])).item()
                        for n, vals in losses.get_summaries()
                    }
                )
            step = str(i).rjust(len(str(max_steps)))
            log_metrics(f"[{epoch}] [{step}/{max_steps}]", l_dict)

            summary_write(
                target_spec_complex.detach(),
                noisy_spec_complex.detach(),
                enh.detach(),
                snrs.detach(),
                lsnr.detach().float(),
                summary_dir,
                prefix=split,
            )


    try:
        cleanup(err, noisy_spec_complex, target_spec_complex, enh, m, noisy_erb_feat, noisy_spec_feat, data)
    except UnboundLocalError as err:
        logger.error(str(err))
    return torch.stack(l_mem).mean().cpu().item()


def setup_losses() -> Loss:
    global state, istft
    assert state is not None

    p = ModelParams()

    istft = Istft(p.fft_size, p.hop_size, torch.as_tensor(state.fft_window().copy())).to(
        get_device()
    )
    loss = Loss(state, istft).to(get_device())
    # loss = torch.jit.script(loss)
    return loss


def load_opt(
    cp_dir: Optional[str], model: nn.Module, mask_only: bool = False, df_only: bool = False
) -> optim.Optimizer:
    lr = config("LR", 5e-4, float, section="optim")
    momentum = config("momentum", 0, float, section="optim")  # For sgd, rmsprop
    decay = config("weight_decay", 0.05, float, section="optim")
    optimizer = config("optimizer", "adamw", str, section="optim").lower()
    betas: Tuple[int, int] = config(
        "opt_betas", [0.9, 0.999], Csv(float), section="optim", save=False  # type: ignore
    )
    if mask_only:
        params = []
        for n, p in model.named_parameters():
            if not ("dfrnn" in n or "df_dec" in n):
                params.append(p)
    elif df_only:
        params = (p for n, p in model.named_parameters() if "df" in n.lower())
    else:
        params = model.parameters()
    supported = {
        "adam": lambda p: optim.Adam(p, lr=lr, weight_decay=decay, betas=betas, amsgrad=True),
        "adamw": lambda p: optim.AdamW(p, lr=lr, weight_decay=decay, betas=betas, amsgrad=True),
        "sgd": lambda p: optim.SGD(p, lr=lr, momentum=momentum, nesterov=True, weight_decay=decay),
        "rmsprop": lambda p: optim.RMSprop(p, lr=lr, momentum=momentum, weight_decay=decay),
    }
    if optimizer not in supported:
        raise ValueError(
            f"Unsupported optimizer: {optimizer}. Must be one of {list(supported.keys())}"
        )
    opt = supported[optimizer](params)
    logger.debug(f"Training with optimizer {opt}")
    if cp_dir is not None:
        try:
            read_cp(opt, "opt", cp_dir, log=False)
        except ValueError as e:
            logger.error(f"Could not load optimizer state: {e}")
    for group in opt.param_groups:
        group.setdefault("initial_lr", lr)
    return opt


def setup_lrs(steps_per_epoch: int) -> np.ndarray:
    lr = config.get("lr", float, "optim")
    num_epochs = config.get("max_epochs", int, "train")
    lr_min = config("lr_min", 1e-6, float, section="optim")
    lr_warmup = config("lr_warmup", 1e-4, float, section="optim")
    assert lr_warmup < lr
    warmup_epochs = config("warmup_epochs", 3, int, section="optim")
    lr_cycle_mul = config("lr_cycle_mul", 1.0, float, section="optim")
    lr_cycle_decay = config("lr_cycle_decay", 0.5, float, section="optim")
    lr_cycle_epochs = config("lr_cycle_epochs", -1, int, section="optim")
    lr_values = cosine_scheduler(
        lr,
        lr_min,
        epochs=num_epochs,
        niter_per_ep=steps_per_epoch,
        warmup_epochs=warmup_epochs,
        start_warmup_value=lr_warmup,
        initial_ep_per_cycle=lr_cycle_epochs,
        cycle_decay=lr_cycle_decay,
        cycle_mul=lr_cycle_mul,
    )
    return lr_values


def setup_wds(steps_per_epoch: int) -> Optional[np.ndarray]:
    decay = config("weight_decay", 0.05, float, section="optim")
    decay_end = config("weight_decay_end", -1, float, section="optim")
    if decay_end == -1:
        return None
    if decay == 0.0:
        decay = 1e-12
        logger.warning("Got 'weight_decay_end' value > 0, but weight_decay is disabled.")
        logger.warning(f"Setting initial weight decay to {decay}.")
        config.overwrite("optim", "weight_decay", decay)
    num_epochs = config.get("max_epochs", int, "train")
    decay_values = cosine_scheduler(
        decay, decay_end, niter_per_ep=steps_per_epoch, epochs=num_epochs
    )
    return decay_values


@torch.no_grad()
def summary_write(
    clean: Tensor,
    noisy: Tensor,
    enh: Tensor,
    snrs: Tensor,
    lsnr: Tensor,
    summary_dir: str,
    prefix="train",
    idx: Optional[int] = None,
):
    global state
    assert state is not None

    p = ModelParams()
    bs = 32

    if idx is None:
        idx = random.randrange(bs)

    snr = snrs[idx]

    def synthesis(x: Tensor) -> Tensor:
        return torch.as_tensor(state.synthesis(make_np(as_complex(x.detach()))))

    #print(clean[0].shape, noisy[0].shape, enh[0].shape)
    torchaudio.save(
        os.path.join(summary_dir, f"{prefix}_clean_snr{snr}.wav"), synthesis(clean[idx]), p.sr
    )
    torchaudio.save(
        os.path.join(summary_dir, f"{prefix}_noisy_snr{snr}.wav"), synthesis(noisy[idx]), p.sr
    )
    torchaudio.save(
        os.path.join(summary_dir, f"{prefix}_enh_snr{snr}.wav"), synthesis(enh[idx]), p.sr
    )
    '''
    np.savetxt(
        os.path.join(summary_dir, f"{prefix}_lsnr_snr{snr}.txt"),
        lsnr[idx].detach().cpu().numpy(),
        fmt="%.3f",
    )
    '''


def summary_noop(*__args, **__kwargs):  # type: ignore
    pass


def get_sigusr1_handler(base_dir):
    def h(*__args):  # type: ignore
        global should_stop
        logger.warning("Received timeout signal. Stopping after current epoch")
        should_stop = True
        continue_file = os.path.join(base_dir, "continue")
        logger.warning(f"Writing {continue_file}")
        open(continue_file, "w").close()

    return h


def cleanup(*args):
    import gc

    for arg in args:
        del arg
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    from icecream import ic, install

    ic.includeContext = True
    install()
    main()

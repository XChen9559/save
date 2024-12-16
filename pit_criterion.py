# Created on 2018/12
# Author: Kaituo XU

from itertools import permutations

import torch
from torch import Tensor, nn
#from utils import angle, as_complex
import torch.nn.functional as F
from typing import Dict, Iterable, List, Optional, Union
from utils import  IMDCT_signal_hop960,device
import librosa

use_mrsl = True

EPS = 1e-8
frame_size = 512
r1=0.35
r2=0.35
r3=0.15
r4=0.1
r5=0.05

class Stft(nn.Module):
    def __init__(self, n_fft: int, hop: Optional[int] = None, window: Optional[Tensor] = None):
        super().__init__()
        self.n_fft = n_fft
        self.hop = hop or n_fft // 4
        if window is not None:
            assert window.shape[0] == n_fft
        else:
            window = torch.hann_window(self.n_fft).to(device)
        self.w: torch.Tensor
        self.register_buffer("w", window)

    def forward(self, input: Tensor):
        # Time-domain input shape: [B, *, T]
        t = input.shape[-1]
        sh = input.shape[:-1]
        out = torch.stft(
            input.reshape(-1, t),
            n_fft=self.n_fft,
            hop_length=self.hop,
            window=self.w,
            normalized=True,
            return_complex=True,
        )
        out = out.view(*sh, *out.shape[-2:])
        return out

class Istft(nn.Module):
    def __init__(self, n_fft_inv: int, hop_inv: int, window_inv: Tensor):
        super().__init__()
        # Synthesis back to time domain
        self.n_fft_inv = n_fft_inv
        self.hop_inv = hop_inv
        self.w_inv: torch.Tensor
        assert window_inv.shape[0] == n_fft_inv
        self.register_buffer("w_inv", window_inv)

    def forward(self, input: Tensor):
        # Input shape: [B, * T, F, (2)]
        input = as_complex(input)
        t, f = input.shape[-2:]
        sh = input.shape[:-2]
        # Even though this is not the DF implementation, it numerical sufficiently close.
        # Pad one extra step at the end to get original signal length
        out = torch.istft(
            F.pad(input.reshape(-1, t, f).transpose(1, 2), (0, 1)),
            n_fft=self.n_fft_inv,
            hop_length=self.hop_inv,
            window=self.w_inv,
            normalized=True,
        )
        if input.ndim > 2:
            out = out.view(*sh, out.shape[-1])
        return out

#[spectralloss]
factor_magnitude = 1000 #1000
factor_complex = 1000 #1000
factor_under = 1

# [multiresspecloss]
mrsl_ffts: List[int] = [128, 256, 512, 1024]
stfts = nn.ModuleDict({str(n_fft): Stft(n_fft) for n_fft in mrsl_ffts})
#gamma = 0.3 #0.3  ;1 : no need to compress power again.
def MultiResSpecLoss(input, target, ri_factor, mag_factor,use_pow_compress):
    #print(input.shape)
    #print(target.shape)
    f_complex = [ri_factor] * len(stfts)
    f = mag_factor
    gamma = 1 if use_pow_compress else 0.3

    loss = torch.zeros((), device=input.device, dtype=input.dtype)
    for i, stft in enumerate(stfts.values()):
        Y = stft.forward(input)
        S = stft.forward(target)
        Y_abs = Y.abs()
        S_abs = S.abs()
        if gamma != 1:
            Y_abs = Y_abs.clamp_min(1e-12).pow(gamma)
            S_abs = S_abs.clamp_min(1e-12).pow(gamma)
        loss += F.mse_loss(Y_abs, S_abs) * f
        if f_complex is not None:
            if gamma != 1:
                Y = Y_abs * torch.exp(1j * angle.apply(Y))
                S = S_abs * torch.exp(1j * angle.apply(S))
            loss += F.mse_loss(torch.view_as_real(Y), torch.view_as_real(S)) * f_complex[i]
    return loss

EPS = 1e-8

def write(inputs, filename, sr=16000):
        librosa.output.write_wav(filename, inputs, sr)# norm=True)

def cal_loss(source, estimate_source):
    """
    Args:
        source: [B, C, T], B is batch size
        estimate_source: [B, C, T]
        source_lengths: [B]
    """
    B, C, T = source.shape
    
    loss_total =0
    mse_loss =0 
    mm,nn=0,0
    for i in range(source.shape[0]):
       if torch.mean(torch.abs(source[i:i+1, :1]))>1e-5:
          max_snr = cal_snr_with_pit(source[i:i+1], estimate_source[i:i+1])
          loss_SDR_PIT = 0 - torch.mean(max_snr)
          loss_total += loss_SDR_PIT
          mm+=1          
       else:
          mse_loss += F.mse_loss(source[i:i+1], estimate_source[i:i+1])
          nn+=1
    #print(mm,nn)
    if (mm>0) & (nn>0):
       loss_all = loss_total/mm + mse_loss/nn
    if (mm>0) & (nn<=0):
       loss_all = loss_total/mm 
    if (mm<=0) & (nn>0):
       loss_all = mse_loss/nn 
    

    return loss_all
    
def cal_snr_with_pit(source, estimate_source):
    """Calculate SI-SNR with PIT training.
    Args:
        source: [B, C, T], B is batch size
        estimate_source: [B, C, T]
        source_lengths: [B], each item is between [0, T]
    """
    #assert source.size() == estimate_source.size()
    source = source[:,:,:estimate_source.shape[2]]
    B, C, T = source.size()
    # mask padding position along T
    #mask = get_mask(source, source_lengths)
    #estimate_source *= mask

    # Step 1. Zero-mean norm
    #num_samples = source_lengths.view(-1, 1, 1).float()  # [B, 1, 1]
    mean_target = torch.sum(source, dim=2, keepdim=True) / T
    mean_estimate = torch.sum(estimate_source, dim=2, keepdim=True) / T
    zero_mean_target = source - mean_target
    zero_mean_estimate = estimate_source - mean_estimate
    # mask padding position along T
    #zero_mean_target *= mask
    #zero_mean_estimate *= mask

    # Step 2. SI-SNR with PIT
    # reshape to use broadcast
    s_target = torch.unsqueeze(zero_mean_target, dim=1)  # [B, 1, C, T]
    s_estimate = torch.unsqueeze(zero_mean_estimate, dim=2)  # [B, C, 1, T]
    # s_target = <s', s>s / ||s||^2
    pair_wise_dot = torch.sum(s_estimate * s_target, dim=3, keepdim=True)  # [B, C, C, 1]
    s_target_energy = torch.sum(s_target ** 2, dim=3, keepdim=True) + EPS  # [B, 1, C, 1]
    pair_wise_proj = pair_wise_dot * s_target / s_target_energy  # [B, C, C, T]

    #pair_wise_proj = s_target

    # e_noise = s' - s_target
    e_noise = s_estimate - s_target  # [B, C, C, T]
    # SI-SNR = 10 * log_10(||s_target||^2 / ||e_noise||^2)
    pair_wise_si_snr = torch.sum(s_target ** 2, dim=3) / (torch.sum(e_noise ** 2, dim=3) + EPS)
    pair_wise_si_snr = 10 * torch.log10(pair_wise_si_snr + EPS)  # [B, C, C]

    
    # Get max_snr of each utterance
    # permutations, [C!, C]
    perms = source.new_tensor(list(permutations(range(C))), dtype=torch.long)
    # one-hot, [C!, C, C]
    index = torch.unsqueeze(perms, 2)
    perms_one_hot = source.new_zeros((*perms.size(), C)).scatter_(2, index, 1)
    # [B, C!] <- [B, C, C] einsum [C!, C, C], SI-SNR sum of each permutation
    snr_set = torch.einsum('bij,pij->bp', [pair_wise_si_snr, perms_one_hot])
    max_snr_idx = torch.argmax(snr_set, dim=1)  # [B]
    # max_snr = torch.gather(snr_set, 1, max_snr_idx.view(-1, 1))  # [B, 1]
    max_snr, _ = torch.max(snr_set, dim=1, keepdim=True)
    max_snr = max_snr/C
    #max_snr = pair_wise_si_snr/C
    
    return max_snr

def cal_si_snr_with_pit(source, estimate_source, source_lengths):
    """Calculate SI-SNR with PIT training.
    Args:
        source: [B, C, T], B is batch size
        estimate_source: [B, C, T]
        source_lengths: [B], each item is between [0, T]
    """
    #assert source.size() == estimate_source.size()
    source = source[:,:,:estimate_source.shape[2]]
    B, C, T = source.size()
    # mask padding position along T
    #mask = get_mask(source, source_lengths)
    #estimate_source *= mask

    # Step 1. Zero-mean norm
    #num_samples = source_lengths.view(-1, 1, 1).float()  # [B, 1, 1]
    mean_target = torch.sum(source, dim=2, keepdim=True) / T
    mean_estimate = torch.sum(estimate_source, dim=2, keepdim=True) / T
    zero_mean_target = source - mean_target
    zero_mean_estimate = estimate_source - mean_estimate
    # mask padding position along T
    zero_mean_target *= mask
    zero_mean_estimate *= mask

    # Step 2. SI-SNR with PIT
    # reshape to use broadcast
    s_target = torch.unsqueeze(zero_mean_target, dim=1)  # [B, 1, C, T]
    s_estimate = torch.unsqueeze(zero_mean_estimate, dim=2)  # [B, C, 1, T]
    # s_target = <s', s>s / ||s||^2
    pair_wise_dot = torch.sum(s_estimate * s_target, dim=3, keepdim=True)  # [B, C, C, 1]
    s_target_energy = torch.sum(s_target ** 2, dim=3, keepdim=True) + EPS  # [B, 1, C, 1]
    pair_wise_proj = pair_wise_dot * s_target / s_target_energy  # [B, C, C, T]

    #pair_wise_proj = s_target

    # e_noise = s' - s_target
    e_noise = s_estimate - s_target  # [B, C, C, T]
    # SI-SNR = 10 * log_10(||s_target||^2 / ||e_noise||^2)
    pair_wise_si_snr = torch.sum(s_target ** 2, dim=3) / (torch.sum(e_noise ** 2, dim=3) + EPS)
    pair_wise_si_snr = 10 * torch.log10(pair_wise_si_snr + EPS)  # [B, C, C]

    # Get max_snr of each utterance
    # permutations, [C!, C]
    perms = source.new_tensor(list(permutations(range(C))), dtype=torch.long)
    # one-hot, [C!, C, C]
    index = torch.unsqueeze(perms, 2)
    perms_one_hot = source.new_zeros((*perms.size(), C)).scatter_(2, index, 1)
    # [B, C!] <- [B, C, C] einsum [C!, C, C], SI-SNR sum of each permutation
    snr_set = torch.einsum('bij,pij->bp', [pair_wise_si_snr, perms_one_hot])
    max_snr_idx = torch.argmax(snr_set, dim=1)  # [B]
    # max_snr = torch.gather(snr_set, 1, max_snr_idx.view(-1, 1))  # [B, 1]
    max_snr, _ = torch.max(snr_set, dim=1, keepdim=True)
    max_snr /= C
    return max_snr, perms, max_snr_idx

def reorder_source(source, perms, max_snr_idx):
    """
    Args:
        source: [B, C, T]
        perms: [C!, C], permutations
        max_snr_idx: [B], each item is between [0, C!)
    Returns:
        reorder_source: [B, C, T]
    """
    B, C, *_ = source.size()
    # [B, C], permutation whose SI-SNR is max of each utterance
    # for each utterance, reorder estimate source according this permutation
    max_snr_perm = torch.index_select(perms, dim=0, index=max_snr_idx)
    # print('max_snr_perm', max_snr_perm)
    # maybe use torch.gather()/index_select()/scatter() to impl this?
    reorder_source = torch.zeros_like(source)
    for b in range(B):
        for c in range(C):
            reorder_source[b, c] = source[b, max_snr_perm[b][c]]
    return reorder_source

def get_mask(source, source_lengths):
    """
    Args:
        source: [B, C, T]
        source_lengths: [B]
    Returns:
        mask: [B, 1, T]
    """
    B, _, T = source.size()
    mask = source.new_ones((B, 1, T))
    for i in range(B):
        mask[i, :, source_lengths[i]:] = 0
    return mask
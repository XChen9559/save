#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 19-10-28 17:41:29
# @Author   : zm
# @File     : train_furca.py
# @Software : PyCharm

import argparse

import torch

from data_mb import AudioDataLoader, AudioDataset
from solver import Solver
#from amazing import FaSNet_base
from models import FaSNet_base
from utils import device

parser = argparse.ArgumentParser(
    "Dual-Path RNN speech separation network with Permutation Invariant Training")
# General config
# Task related
parser.add_argument('--train_dir', type=str, default='/common-data/mengruijie/data/wsj-2mix-16k/json/tr/', #
                    help='directory including mix.json, s1.json and s2.json')
parser.add_argument('--valid_dir', type=str, default='/common-data/mengruijie/data/wsj-2mix-16k/json/cv/',
                    help='directory including mix.json, s1.json and s2.json')
parser.add_argument('--sample_rate', default=48000, type=int,
                    help='Sample rate')
parser.add_argument('--segment', default=4, type=float,
                    help='Segment length (seconds)')
parser.add_argument('--cv_maxlen', default=12, type=float,
                    help='max audio length (seconds) in cv, to avoid OOM issue.')
# Network architecture
parser.add_argument('--N', default=64, type=int,
                    help='Dim of feature to the DPRNN')
parser.add_argument('--W', default=2, type=int,
                    help='Filter lenght in encoder, or the length of window in samples')
parser.add_argument('--K', default=250, type=int,
                    help='Chunk size in frames')
parser.add_argument('--D', default=6, type=int,
                    help='Number of DPRNN blocks')
parser.add_argument('--C', default=2, type=int,
                    help='Number of speakers')
parser.add_argument('--E', default=256, type=int,
                    help='Number of channels in bottleneck 1 × 1-conv block, dim of feature to the DPRNN')
parser.add_argument('--H', default=128, type=int,
                    help='Number of hidden units in each direction of RNN')
parser.add_argument('--norm_type', default='gLN', type=str,
                    choices=['gLN', 'cLN', 'BN'], help='Layer norm type')
parser.add_argument('--causal', type=int, default=0,
                    help='Causal (1) or noncausal(0) training')
parser.add_argument('--mask_nonlinear', default='relu', type=str,
                    choices=['relu', 'softmax'], help='non-linear to generate mask')
# Training config
parser.add_argument('--use_cuda', type=int, default=1,
                    help='Whether use GPU')
parser.add_argument('--epochs', default=100, type=int,
                    help='Number of maximum epochs')
parser.add_argument('--half_lr', dest='half_lr', default=0, type=int,
                    help='Halving learning rate when get small improvement')
parser.add_argument('--early_stop', dest='early_stop', default=0, type=int,
                    help='Early stop training when no improvement for 10 epochs')
parser.add_argument('--max_norm', default=5, type=float,
                    help='Gradient norm threshold to clip')
# minibatch
parser.add_argument('--shuffle', default=1, type=int,
                    help='reshuffle the data at every epoch')
parser.add_argument('--batch_size', default=4, type=int,
                    help='Batch size')
# optimizer
parser.add_argument('--optimizer', default='adam', type=str,
                    choices=['sgd', 'adam'],
                    help='Optimizer (support sgd and adam now)')
parser.add_argument('--lr', default=0.00001, type=float,
                    help='Init learning rate')
parser.add_argument('--momentum', default=0.0, type=float,
                    help='Momentum for optimizer')
parser.add_argument('--l2', default=0.0, type=float,
                    help='weight decay (L2 penalty)')
# save and load model
parser.add_argument('--save_folder', default='exp1/',
                    help='Location to save epoch models')
parser.add_argument('--checkpoint', dest='checkpoint', default=1, type=int,
                    help='Enables checkpoint saving of model')
parser.add_argument('--continue_from', default='exp1/temp_best.pth.tar',
                    help='Continue from checkpoint model')
parser.add_argument('--model_path', default='exp1/',
                    help='Location to save best validation model')
# logging
parser.add_argument('--print_freq', default=100, type=int,
                    help='Frequency of printing training infomation')
#parser.add_argument('--csv_path', default='../Libri_ID_listbig.csv',
#                    help='wsj0 ID list csv file')
#parser.add_argument('--ref_duration', default=0, type=float,
#                    help='Reference wave length (second)')
                    
def testFlops(model):
    from thop import profile
    from thop import clever_format
    print("<<Test Flops Begin>>")
    #input.shape torch.Size([1, 252, 512])  4s?
    #embeddingVector.shape torch.Size([1, 128])
    x1 = torch.rand(1, 63, 512)
    #x2 = torch.rand(1, 128).cuda()  #embeddingVector
    h_list = []
    c_list = []
    for i_list in range(6):
        h = torch.zeros([2, 34,48]).type(x1.type())
        c = torch.zeros([2, 34,48]).type(x1.type())
        h_list.append(h)
        c_list.append(c)
    y, _, _ = model(x1, h_list, c_list)
    
    macs, params = profile(model, inputs=(x1, h_list, c_list))
    macs, params = clever_format([macs, params], "%.3f")
    print(macs)
    print(params)
    print("<<Test Flops End>>")

def main(args):
    # Construct Solver
    # data
    tr_dataset = AudioDataset(args.train_dir, args.batch_size, 
                              sample_rate=args.sample_rate, segment=args.segment,flag_eval=False)
    cv_dataset = AudioDataset(args.valid_dir, batch_size=args.batch_size,  # 1 -> use less GPU memory to do cv                              
                              sample_rate=args.sample_rate,
                              segment=-1, cv_maxlen=args.cv_maxlen,flag_eval=True)  # -1 -> use full audio
    tr_loader = AudioDataLoader(tr_dataset, batch_size=1,
                                shuffle=args.shuffle)
    cv_loader = AudioDataLoader(cv_dataset, batch_size=1)
    data = {'tr_loader': tr_loader, 'cv_loader': cv_loader}

    frame_size = 1536
    rfft_size = frame_size//2

    # model
    # model = FURCA(args.W, args.N, args.K, args.C, args.D, args.H, args.E,
    #                    norm_type=args.norm_type, causal=args.causal,
    #                    mask_nonlinear=args.mask_nonlinear)
    #model = FaSNet_base(enc_dim=256, feature_dim=64, hidden_dim=128, layer=6, segment_size=250, nspk = 2, win_len = 2)
    #model = FaSNet_base(enc_dim=32, feature_dim=16, hidden_dim=16, layer=2, segment_size=250, nspk=2, win_len=2)
    model = FaSNet_base(enc_dim=96, feature_dim=64, hidden_dim=64, layer=5, group_size=40, segment_size=1024//2, trunk_size=1, nspk = 2, win_len = 2).cuda()
    

    # Find total parameters and trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')

    if args.use_cuda:
        # model = torch.nn.DataParallel(model)
        if torch.cuda.device_count() > 1:
            print("Use ",torch.cuda.device_count()," GPUs!")
            available_gpus = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]
            print(available_gpus)
            # model = torch.nn.DataParallel(model, device_ids=[0,1,2,3,4,5,6,7])
            model = torch.nn.DataParallel(model).cuda()
        else:
            model.cuda()

    # optimizer
    if args.optimizer == 'sgd':
        optimizier = torch.optim.SGD(model.parameters(),
                                     lr=args.lr,
                                     momentum=args.momentum,
                                     weight_decay=args.l2)
    elif args.optimizer == 'adam':
        optimizier = torch.optim.Adam(model.parameters(),
                                      lr=args.lr,
                                      weight_decay=args.l2)
    else:
        print("Not support optimizer")
        return

    # solver
    solver = Solver(data, model, optimizier, args, frame_size)
    solver.train()


if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    main(args)




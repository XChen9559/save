#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time     : 19-11-1 09:12:12
# @Author   : zm
# @File     : model.py
# @Software : PyCharm

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy
from numpy.lib import stride_tricks
from utils import overlap_and_add, device
from torch.autograd import Variable
#import ipdb

EPS = 1e-8

        

class Encoder(nn.Module):
    """Estimation of the nonnegative mixture weight by a 1-D conv layer.
    """
    def __init__(self, W=2, N=64, seg_size=129):
        super(Encoder, self).__init__()
        # Hyper-parameter
        self.W, self.N = W, N
        self.seg_size = seg_size
        # Components
        # 50% overlap
        #self.conv1d_U = nn.Conv1d(1, N, kernel_size=W, stride=W // 2, bias=False)
        self.conv1d_U_Subband1 = nn.Conv1d(1, N, kernel_size=8, stride=4, padding=0, bias=False)        # 2b*8 = 16b: 0~15
        self.conv1d_U_Subband2 = nn.Conv1d(1, N, kernel_size=12, stride=8, padding=0, bias=False)       # 4b*8 = 32b: 16~47
        self.conv1d_U_Subband3 = nn.Conv1d(1, N, kernel_size=20, stride=16, padding=0, bias=False)      # 8b*10 = 80b: 48~127
        self.conv1d_U_Subband4 = nn.Conv1d(1, N, kernel_size=36, stride=32, padding=0, bias=False)      # 16b*8 = 128b: 128~255                
        self.conv1d_U_Subband5 = nn.Conv1d(1, N, kernel_size=64, stride=64, padding=0, bias=False)      #16b*8 = 128b: 256~384  
        self.conv1d_U_Subband6 = nn.Conv1d(1, N, kernel_size=128, stride=128, padding=0, bias=False)      # 32b*4 = 256b: 384~512
        
                
    def forward(self, mixture):
        """
        Args:
            mixture: [B, T], B is batch size, T is #samples
        Returns:
            mixture_w: [B, N, L], where L = (T-W)/(W/2)+1 = 2T/W-1
            L is the number of time steps
        """
        batch_size = mixture.shape[0]
        mixture = torch.unsqueeze(mixture.view(-1, self.seg_size*2), 1)  # [B, 1, T]
        mixture_subband1 = torch.cat([torch.zeros(mixture.shape[0], mixture.shape[1], 2).type(mixture.type()), mixture[: ,: ,0:34]], axis=2) ## 8
        mixture_subband2 = mixture[: ,: ,30:98]
        mixture_subband3 = mixture[:, :, 94:258] ##-4k  160 = 10*160
        mixture_subband4 = mixture[:, :, 254:514] ##4-8k  256=8*32
        mixture_subband5 = mixture[:, :, 512:768] ##8-12k # 256 = 128*2   8--》2       
        mixture_subband6 = torch.cat([mixture[:, :, 766:1024], torch.zeros(mixture.shape[0], mixture.shape[1], 2).type(mixture.type())], axis=2) ## 256 = 64*4   1
        #print(mixture_subband1.shape,mixture_subband2.shape,mixture_subband3.shape,mixture_subband4.shape)
        mixture_w_subband1 = F.relu(self.conv1d_U_Subband1(mixture_subband1))  # [B, N, 8]
        mixture_w_subband2 = F.relu(self.conv1d_U_Subband2(mixture_subband2))  # [B, N, 8]
        mixture_w_subband3 = F.relu(self.conv1d_U_Subband3(mixture_subband3))  # [B, N, 10]
        mixture_w_subband4 = F.relu(self.conv1d_U_Subband4(mixture_subband4))  # [B, N, 8]
        mixture_w_subband5 = F.relu(self.conv1d_U_Subband5(mixture_subband5))  # [B, N, 8]
        mixture_w_subband6 = F.relu(self.conv1d_U_Subband6(mixture_subband6))  # [B, N, 4]
        mixture_w = torch.cat([mixture_w_subband1, mixture_w_subband2, mixture_w_subband3, mixture_w_subband4, mixture_w_subband5, mixture_w_subband6], axis=2)
        #print(mixture_w_subband1.shape,mixture_w_subband2.shape,mixture_w_subband3.shape,mixture_w_subband4.shape,mixture_w_subband5.shape,mixture_w_subband6.shape)
        #print(mixture_w.shape)
        mixture_w = mixture_w
        mixture_w = mixture_w.permute(0,2,1).contiguous().view(batch_size, -1, self.N).permute(0,2,1).contiguous()
        
        #mixture_z = self.rnn(mixture_w.permute(0, 2, 1))
        #mixture_w = F.relu(mixture_z.permute(0, 2, 1))  # [B, N, L]

        return mixture_w

class Decoder(nn.Module):
    def __init__(self, E, W):
        super(Decoder, self).__init__()
        # Hyper-parameter
        self.E, self.W = E, W
        # Components
        #self.basis_signals = nn.Linear(E, 2, bias=False)
        self.basis_signals_subband1 = nn.Linear(E, 4, bias=False)
        self.basis_signals_subband2 = nn.Linear(E, 8, bias=False)
        self.basis_signals_subband3 = nn.Linear(E, 16, bias=False)
        self.basis_signals_subband4 = nn.Linear(E, 32, bias=False)
        self.basis_signals_subband5 = nn.Linear(E, 64, bias=False)
        self.basis_signals_subband6 = nn.Linear(E, 128, bias=False)

    def forward(self, mixture_w, est_mask):
        """
        Args:
            mixture_w: [B, E, L]
            est_mask: [B, C, E, L]
        Returns:
            est_source: [B, C, T]
        """
        # D = W * M
        #print(mixture_w.shape)
        #print(est_mask.shape)
        source_w = torch.unsqueeze(mixture_w, 1) * est_mask  # [B, C, E, L]

        #source_w = torch.unsqueeze(mixture_w, 1) + est_mask  # [B, C, E, L]
        #source_w = est_mask  # [B, C, E, L]
        source_w = torch.transpose(source_w, 2, 3) # [B, C, L, E]
        s = source_w.shape
        source_w = source_w.view(s[0], s[1], -1, 40, s[3])
        source_w_subband1 = source_w[:,:,:,0:8,:]
        source_w_subband2 = source_w[:, :, :, 8:16, :]
        source_w_subband3 = source_w[:, :, :, 16:26, :]
        source_w_subband4 = source_w[:, :, :, 26:34, :]
        source_w_subband5 = source_w[:, :, :, 34:38, :]
        source_w_subband6 = source_w[:, :, :, 38:40, :]

        # S = DV
        #est_source = self.basis_signals(source_w)  # [B, C, L, W]
        est_source_subband1 = self.basis_signals_subband1(source_w_subband1)  # [B, C, L, W]
        est_source_subband2 = self.basis_signals_subband2(source_w_subband2)  # [B, C, L, W]
        est_source_subband3 = self.basis_signals_subband3(source_w_subband3)  # [B, C, L, W]
        est_source_subband4 = self.basis_signals_subband4(source_w_subband4)  # [B, C, L, W]
        est_source_subband5 = self.basis_signals_subband5(source_w_subband5)  # [B, C, L, W]
        est_source_subband6 = self.basis_signals_subband6(source_w_subband6)  # [B, C, L, W]
        
        s = est_source_subband1.shape
        est_source_subband1 = est_source_subband1.view(s[0], s[1], s[2], -1)
        est_source_subband2 = est_source_subband2.view(s[0], s[1], s[2], -1)
        est_source_subband3 = est_source_subband3.view(s[0], s[1], s[2], -1)
        est_source_subband4 = est_source_subband4.view(s[0], s[1], s[2], -1)
        est_source_subband5 = est_source_subband5.view(s[0], s[1], s[2], -1)
        est_source_subband6 = est_source_subband6.view(s[0], s[1], s[2], -1)
        est_source = torch.cat([est_source_subband1, est_source_subband2, est_source_subband3, est_source_subband4,est_source_subband5,est_source_subband6], axis=3)
        #print(est_source_subband1.shape, est_source_subband2.shape, est_source_subband3.shape, est_source_subband4.shape,est_source_subband5.shape,est_source_subband6.shape)
        #print(est_source_subband1.shape[3]+ est_source_subband2.shape[3]+  est_source_subband3.shape[3]+ est_source_subband4.shape[3]+ est_source_subband5.shape[3]+ est_source_subband6.shape[3])
        #est_source = est_source.view(s[0], s[1], -1, 256)
        s = est_source.shape
        est_source = est_source.view(s[0], s[1], -1)
        #est_source = overlap_and_add(est_source, self.W//2) # B x C x T
        #est_source = overlap_and_add(est_source, 1)  # B x C x T
        return est_source
class ColSingleRNN(nn.Module):
    """
    Container module for a single RNN layer.

    args:
        rnn_type: string, select from 'RNN', 'LSTM' and 'GRU'.
        input_size: int, dimension of the input feature. The input should have shape
                    (batch, seq_len, input_size).
        hidden_size: int, dimension of the hidden state.
        dropout: float, dropout ratio. Default is 0.
        bidirectional: bool, whether the RNN layers are bidirectional. Default is False.
    """

    def __init__(self, rnn_type, input_size, hidden_size, dropout=0, bidirectional=False):
        super(ColSingleRNN, self).__init__()

        self.rnn_type = rnn_type
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_direction = int(bidirectional) + 1

        #self.h = torch.zeros([self.num_direction, 250, self.input_size])
        #self.c = torch.zeros([self.num_direction, 250, self.input_size])

        self.rnn = getattr(nn, rnn_type)(input_size, hidden_size, 1, dropout=dropout, batch_first=True,
                                         bidirectional=bidirectional)

        # linear projection layer
        self.proj = nn.Linear(hidden_size * self.num_direction, input_size)

    def forward(self, input, h):
        # input shape: batch, seq, dim
        #input = input.to(device)
        output = input
        #ipdb.set_trace()
        rnn_output, h = self.rnn(output, h)
        rnn_output = self.proj(rnn_output.contiguous().view(-1, rnn_output.shape[2])).view(output.shape)
        return rnn_output, h
        
class SingleRNN(nn.Module):
    """
    Container module for a single RNN layer.

    args:
        rnn_type: string, select from 'RNN', 'LSTM' and 'GRU'.
        input_size: int, dimension of the input feature. The input should have shape
                    (batch, seq_len, input_size).
        hidden_size: int, dimension of the hidden state.
        dropout: float, dropout ratio. Default is 0.
        bidirectional: bool, whether the RNN layers are bidirectional. Default is False.
    """

    def __init__(self, rnn_type, input_size, hidden_size, dropout=0, bidirectional=False):
        super(SingleRNN, self).__init__()

        self.rnn_type = rnn_type
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_direction = int(bidirectional) + 1

        self.rnn = getattr(nn, rnn_type)(input_size, hidden_size, 1, dropout=dropout, batch_first=True,
                                         bidirectional=bidirectional)

        # linear projection layer
        self.proj = nn.Linear(hidden_size * self.num_direction, input_size)

    def forward(self, input):
        # input shape: batch, seq, dim
        #input = input.to(device)
        output = input
        rnn_output, _ = self.rnn(output)
        rnn_output = self.proj(rnn_output.contiguous().view(-1, rnn_output.shape[2])).view(output.shape)
        return rnn_output


# dual-path RNN
class DPRNN(nn.Module):
    """
    Deep duaL-path RNN.

    args:
        rnn_type: string, select from 'RNN', 'LSTM' and 'GRU'.
        input_size: int, dimension of the input feature. The input should have shape
                    (batch, seq_len, input_size).
        hidden_size: int, dimension of the hidden state.
        output_size: int, dimension of the output size.
        dropout: float, dropout ratio. Default is 0.
        num_layers: int, number of stacked RNN layers. Default is 1.
        bidirectional: bool, whether the RNN layers are bidirectional. Default is False.
    """

    def __init__(self, rnn_type, input_size, hidden_size, output_size,trunk_size,
                 dropout=0, num_layers=1, bidirectional=True):
        super(DPRNN, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.trunk_size=trunk_size
        # dual-path RNN
        self.trunk_rnn = nn.ModuleList([])
        self.row_rnn = nn.ModuleList([])
        self.col_rnn = nn.ModuleList([])
       
        self.proj_col = nn.ModuleList([])
        #self.row_norm = nn.ModuleList([])
        #self.col_norm = nn.ModuleList([])
        for i in range(num_layers):
            self.row_rnn.append(SingleRNN('LSTM', input_size, hidden_size, dropout,
                                          bidirectional=True))  # intra-segment RNN is always noncausal
            self.col_rnn.append(ColSingleRNN('GRU', input_size, hidden_size, dropout, bidirectional=False))
            #self.proj_col.append(nn.Linear(hidden_size*2,hidden_size))

        # output layer
        self.output = nn.Sequential(nn.ReLU(),
                                    nn.Conv2d(input_size, output_size, 1)
                                    )

    def forward(self, input, h_list):
        # input shape: batch, N, dim1, dim2
        # apply RNN on dim1 first and then dim2
        # output shape: B, output_size, dim1, dim2
        #input = input.to(device)
        batch_size, N, dim1, dim2 = input.shape
        #print(input.shape)
        output = input
        j = 0
        h_list1 = torch.zeros_like(h_list).type(h_list.type())   
        for i in range(len(self.row_rnn)):
            row_input = output.permute(0, 3, 2, 1).contiguous().view(batch_size*dim2, dim1, N)  # B*dim2, dim1, N
            row_output = self.row_rnn[i](row_input)  # B*dim2, dim1, H
            row_output = row_output.transpose(1,2)
            #row_output = self.row_norm[i](row_output)
            row_output = row_output.transpose(1, 2)
            row_output = row_output.view(batch_size, dim2, dim1, -1).permute(0, 3, 2,
                                                                             1).contiguous()  # B, N, dim1, dim2
            output = output + row_output
        
            col_input = output.permute(0, 2, 3, 1).contiguous().view(batch_size*dim1,dim2, -1)  # B*dim1, dim2, N            
            col_output = torch.zeros([batch_size*dim1, dim2,self.input_size]).type(col_input.type())
            h=h_list[i]                        
            col_output, h = self.col_rnn[i](col_input,h)  # B*dim1, dim2, H
            h_list1[i]=h  
            col_output = col_output.reshape(batch_size, dim1, dim2, -1).permute(0, 3, 1,
                                                                                2).contiguous()  # B, N, dim1, dim2
            output = output + col_output
            
        output = self.output(output) # B, output_size, dim1, dim2

        return output, h_list1

 



# base module for deep DPRNN
class DPRNN_base(nn.Module):
    def __init__(self, input_dim, feature_dim, hidden_dim, num_spk=2,
                 layer=4, group_size=100, trunk_size = 12, bidirectional=True, rnn_type='LSTM'):
        super(DPRNN_base, self).__init__()

        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim

        self.layer = layer
        self.group_size = group_size
        self.trunk_size = trunk_size
        self.num_spk = num_spk

        self.eps = 1e-8

        # bottleneck
        self.BN = nn.Conv1d(self.input_dim, self.feature_dim, 1, bias=False)

        # DPRNN model
        self.DPRNN = DPRNN(rnn_type, self.feature_dim, self.hidden_dim,
                                   self.feature_dim * self.num_spk,self.trunk_size,
                                   num_layers=layer, bidirectional=bidirectional)

    def pad_segment(self, input, group_size, trunk_size):
        # input is the features: (B, N, T)
        batch_size, dim, seq_len = input.shape
        #segment_stride = group_size // 2
        timelength = seq_len//group_size
        rest = timelength - timelength // trunk_size* trunk_size  #group_size*chunk_size - seq_len % (group_size*chunk_size)
        #print('1111111111111',rest,trunk_size,timelength,input.shape)
        if rest > 0:
            pad = Variable(torch.zeros(batch_size, dim, group_size*(trunk_size - rest))).type(input.type())
            input = torch.cat([input, pad], 2)

        #pad_aux = Variable(torch.zeros(batch_size, dim, segment_stride)).type(input.type())
        #input = torch.cat([pad_aux, input, pad_aux], 2)

        return input, rest

    def split_feature(self, input, group_size, trunk_size):
        # split the feature into chunks of segment size
        # input is the features: (B, N, T)

        input, rest = self.pad_segment(input, group_size, trunk_size)
        batch_size, dim, seq_len = input.shape
        #segment_stride = group_size // 2

        #segments1 = input[:, :, :-segment_stride].contiguous().view(batch_size, dim, -1, group_size)
        #segments2 = input[:, :, segment_stride:].contiguous().view(batch_size, dim, -1, group_size)
        #segments = torch.cat([segments1, segments2], 3).view(batch_size, dim, -1, group_size).transpose(2, 3)
        #print('32222222222',input.shape,rest)
        segments = input.contiguous().view(batch_size, dim, -1, group_size).transpose(2, 3)

        return segments.contiguous(), rest

    def merge_feature(self, input, rest,trunk_size):
        # merge the splitted features into full utterance
        # input is the features: (B, N, L, K)

        batch_size, dim, group_size, _ = input.shape
        #segment_stride = group_size // 2
        #input = input.transpose(2, 3).contiguous().view(batch_size, dim, -1, group_size * 2)  # B, N, K, L

        #input1 = input[:, :, :, :group_size].contiguous().view(batch_size, dim, -1)[:, :, segment_stride:]
        #input2 = input[:, :, :, group_size:].contiguous().view(batch_size, dim, -1)[:, :, :-segment_stride]

        #output = input1 + input2
    
        output = input.transpose(2,3).contiguous().view(batch_size, dim, -1)
        if rest > 0:
            output = output[:, :, :-(trunk_size-rest)*group_size]

        return output.contiguous()  # B, N, T

    def forward(self, input, h_list):
        pass

# DPRNN for beamforming filter estimation
class BF_module(DPRNN_base):
    def __init__(self, *args, **kwargs):
        super(BF_module, self).__init__(*args, **kwargs)

        # gated output layer
        self.output = nn.Sequential(nn.Conv1d(self.feature_dim, self.feature_dim, 1),
                                    nn.Tanh()
                                    )
        self.output_gate = nn.Sequential(nn.Conv1d(self.feature_dim, self.feature_dim, 1),
                                         nn.Sigmoid()
                                         #nn.ReLU()
                                         )

    def forward(self, input, h_list):
        #input = input.to(device)
        # input: (B, E, T)
        batch_size, E, seq_length = input.shape

        enc_feature = self.BN(input) # (B, E, L)-->(B, N, L)
        # split the encoder output into overlapped, longer segments
        enc_segments, enc_rest = self.split_feature(enc_feature, self.group_size, self.trunk_size)  # B, N, L, K: L is the group_size
        #print('enc_segments.shape {}'.format(enc_segments.shape))
        # pass to DPRNN

        output, h_list1 = self.DPRNN(enc_segments, h_list)
        output = output.view(batch_size * self.num_spk, self.feature_dim, self.group_size,
                                                   -1)  # B*nspk, N, L, K

        # overlap-and-add of the outputs
        output = self.merge_feature(output, enc_rest, self.trunk_size)  # B*nspk, N, T

        # gated output layer for filter generation
        bf_filter = self.output(output) * self.output_gate(output)  # B*nspk, K, T
        bf_filter = bf_filter.transpose(1, 2).contiguous().view(batch_size, self.num_spk, -1,
                                                                self.feature_dim)  # B, nspk, T, N

        return bf_filter, h_list1


# base module for FaSNet
class FaSNet_base(nn.Module):
    def __init__(self, enc_dim, feature_dim, hidden_dim, layer, group_size=34, segment_size=250, trunk_size=12,
                 nspk=2, win_len=2):
        super(FaSNet_base, self).__init__()

        # parameters
        self.window = win_len
        self.stride = self.window // 2

        self.enc_dim = enc_dim
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.segment_size = segment_size
        self.group_size = group_size
        self.trunk_size = trunk_size
        self.layer = layer
        self.num_spk = nspk
        self.eps = 1e-8

        # waveform encoder
        #self.encoder = nn.Conv1d(1, self.enc_dim, self.feature_dim, bias=False)
        self.encoder = Encoder(win_len, enc_dim, segment_size) # [B T]-->[B N L]
        
        #self.enc_LN = nn.GroupNorm(1, self.enc_dim, eps=1e-8) # [B N L]-->[B N L]
        self.separator = BF_module(self.enc_dim, self.feature_dim, self.hidden_dim,
                                self.num_spk, self.layer, self.group_size,self.trunk_size)
        # [B, N, L] -> [B, E, L]
        self.mask_conv1x1 = nn.Conv1d(self.feature_dim, self.enc_dim, 1, bias=False)
        #self.decoder = Decoder(enc_dim, win_len)
        self.decoder1 = Decoder(enc_dim, 1)
        self.decoder2 = Decoder(enc_dim, 1)

    def pad_input(self, input, window):
        """
        Zero-padding input according to window/stride size.
        """
        batch_size, nsample = input.shape
        stride = window // 2

        # pad the signals at the end for matching the window/stride size
        rest = window - (stride + nsample % window) % window
        if rest > 0:
            pad = torch.zeros(batch_size, rest).type(input.type())
            input = torch.cat([input, pad], 1)
        pad_aux = torch.zeros(batch_size, stride).type(input.type())
        input = torch.cat([pad_aux, input, pad_aux], 1)

        return input, rest

    def forward(self, input, h_list):
        """
        input: shape (batch, T)
        """
        # pass to a DPRNN
        #input = input.to(device)
        B, FN, _ = input.size()
        #print(input.shape)
        #print(input.shape)
        # mixture, rest = self.pad_input(input, self.window)
        #print('mixture.shape {}'.format(mixture.shape))
       
        mixture_w = self.encoder(input)  # B, E, L
        #print('mixture_w.shape {}'.format(mixture_w.shape))
        #score_ = self.enc_LN(mixture_w) # B, E, L
        #print('mixture_w.shape {}'.format(mixture_w.shape))
        score_, h_list1 = self.separator(mixture_w, h_list)  # B, nspk, T, N
        #print('score_.shape {}'.format(score_.shape))
        score_ = score_.view(B*self.num_spk, -1, self.feature_dim).transpose(1, 2).contiguous()  # B*nspk, N, T
        #print('score_.shape {}'.format(score_.shape))
        score = self.mask_conv1x1(score_)  # [B*nspk, N, L] -> [B*nspk, E, L]
        #print('score.shape {}'.format(score.shape))
        score = score.view(B, self.num_spk, self.enc_dim, -1)  # [B*nspk, E, L] -> [B, nspk, E, L]
        #print('score.shape {}'.format(score.shape))
        est_mask = F.relu(score)

        #print(mixture_w.shape)
        #print(est_mask.shape)
        
        est_source1 = self.decoder1(mixture_w, est_mask[:,0:1]) # [B, E, L] + [B, nspk, E, L]--> [B, nspk, T]
        est_source2 = self.decoder2(mixture_w, est_mask[:,1:]) # [B, E, L] + [B, nspk, E, L]--> [B, nspk, T]
        est_source = torch.cat([est_source1,est_source2],dim=1)

        # if rest > 0:
        #     est_source = est_source[:, :, :-rest]

        return est_source, h_list1
        
if __name__=='__main__':
    from thop import profile
    from thop import clever_format
    print("<<Test Flops Begin>>")
    #1.155G
    #622.720K

    #
    #input.shape torch.Size([1, 252, 512])  4s?
    #embeddingVector.shape torch.Size([1, 128])
    batch=1
    x1 = torch.rand(batch, 50, 1024).cuda()
    #x2 = torch.rand(1, 128).cuda()  #embeddingVector
#    h_list = []
#    for i_list in range(5):
#        h = torch.zeros([1, batch*40,64]).type(x1.type())
#        h_list.append(h)
    h_list = torch.zeros([5,1, batch*40,64]).type(x1.type()) 
    rfft_size = 512
    model = FaSNet_base(enc_dim=96, feature_dim=64, hidden_dim=64, layer=5, group_size=40, segment_size=1024//2, trunk_size=1, nspk = 2, win_len = 2).cuda()
    
    y, _ = model(x1, h_list)
    print(y.shape)
    macs, params = profile(model, inputs=(x1, h_list, ))
    macs, params = clever_format([macs, params], "%.3f")
    print(macs)
    print(params)
    print("<<Test Flops End>>")
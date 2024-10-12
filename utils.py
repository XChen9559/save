# Created on 2018/12
# Author: Kaituo XU

import math

import torch
from torch import Tensor
#import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Function
pyv = '180'  #'041'  '180   '

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def freq_to_mel(f,nk):
    return 1127.01048*np.log(f*2**(nk/12)/700+1)

def mel_to_req(f_mel,nk):
    return  (10**(f_mel/1127.01048)-1)*700/(2**(nk/12))
        
def mel_octal(fs,fft_size,nbins):
    #fmin=0
    #fmax=fs//2
    fmin=0
    fmax = fs//2
    f_mel_min = freq_to_mel(fmin,0)
    f_mel_max = freq_to_mel(fmax,nbins)
    
    mel_feat=torch.zeros([fft_size,nbins])
    f_mel_nb = torch.linspace(f_mel_min,f_mel_max,nbins+2)  ###均匀分组在mel刻度上
    #f_nb = mel_to_req(f_mel_nb)
    #print(f_mel_nb.shape)
    for i in range(1,nbins+1):
        f0=f_mel_nb[i]
        #n1 = int(f_nb[i-1]/fs*fft_size)
        #n2 = int(f_nb[i+1]/fs*fft_size)
        n1 =int(mel_to_req(f_mel_nb[i-1],i))
        n2 =int(mel_to_req(f_mel_nb[i+1],i))
        mel_feat[n1:n2,i-1]=1  ##可替换成三角函数
    return mel_feat
    
    
win_hop960 = torch.sin(torch.Tensor(np.pi * np.array(range(576)) / (576 - 1)) / 2)
win_hop960 = torch.cat([win_hop960, torch.ones(1536 - 2 * 576), win_hop960.flip(0)], 0)
win_hop960 = win_hop960.to(device)

def MDCT_signal_hop960(input, frame_size, use_power_compress=False):
    '''
    output [B, Frms, fft_n]
    '''
    batch_size, nsample = input.shape
    stride = 960
    rfft_size = frame_size // 2+1
    overlap = frame_size-stride

    # pad the signals at the end for matching the window/stride size
    rest = stride - nsample % stride
    if rest > 0:
        pad = torch.zeros(batch_size, rest).type(input.type())
        input = torch.cat([input, pad], 1)
    pad_aux1 = torch.zeros(batch_size, overlap).type(input.type())
    pad_aux2 = torch.zeros(batch_size, stride).type(input.type())
    input = torch.cat([pad_aux1, input, pad_aux2], 1).to(device)
    segments1 = input[:, :-overlap].contiguous().view(batch_size, -1, stride)
    segments2 = segments1[:,1:, :overlap]
    segments = torch.cat([segments1[:,:-1], segments2], 2).view(batch_size, -1, frame_size)
    if pyv == '180':
        segments = torch.fft.rfft(segments * win_hop960, frame_size, 2)
        sh = segments.shape
        segments = torch.cat(
            [torch.real(segments).view(sh[0], sh[1], sh[2], 1), torch.imag(segments).view(sh[0], sh[1], sh[2], 1)], 3)
    else:
        segments = torch.rfft(segments * win_hop960, 1)
    segments = segments.view(batch_size, -1, rfft_size*2)
    segments = segments[:,:,:-2]

    if use_power_compress:
        s_tmp = segments.shape
        segments = segments.view(s_tmp[0], s_tmp[1], -1, 2) #[4, 252, 256, 2])
        segments = power_compress(segments)  # [b,frms, n_fft(512)/2,2]
        segments = segments.view(s_tmp[0], s_tmp[1], -1) #[8, 252, 512]
    return segments
    
def IMDCT_signal_hop960(est_output, frame_size, use_power_compress=False):
    '''
    output [B, T]
    '''
    stride = 960
    FB_num = frame_size // 2
    overlap = frame_size - stride
    bs = est_output.shape[0]
    s1_FFT_output = est_output.view(bs, -1, FB_num, 2)  # [4, 252, 256, 2]
    if use_power_compress:
        s1_FFT_output = power_uncompress(s1_FFT_output[:,:,:,0], s1_FFT_output[:,:,:,1])

    pad_aux = torch.zeros(bs, s1_FFT_output.shape[1], 1, 2).type(est_output.type())
    s1_FFT_output = torch.cat([s1_FFT_output, pad_aux], 2)
    if pyv == '180':
        s1_FFT_output = torch.complex(s1_FFT_output[:, :, :, 0], s1_FFT_output[:, :, :, 1])
        s1_output = torch.fft.irfft(s1_FFT_output, frame_size, 2) * win_hop960 #[4, 252, 512] [4, 201, 512])320win
    else:
        s1_output = torch.irfft(s1_FFT_output, 1, signal_sizes=[frame_size]) * win_hop960

    tmp = s1_output[:,:-1,overlap:]
    tmp[:,:,-overlap:] = s1_output[:,:-1,-overlap:]+s1_output[:,1:,:overlap]
    s1_output = tmp.reshape(bs, 1, -1)  #[4, 1, 64000]
    return s1_output
    
def MDCT_signal(input, frame_size):
    batch_size, nsample = input.shape
    stride = frame_size // 2
    rfft_size = stride+1

    # pad the signals at the end for matching the window/stride size
    rest = frame_size - (stride + nsample % frame_size) % frame_size
    if rest > 0:
        pad = torch.zeros(batch_size, rest).type(input.type())
        input = torch.cat([input, pad], 1)
    pad_aux = torch.zeros(batch_size, stride).type(input.type())
    input = torch.cat([pad_aux, input, pad_aux], 1)

    segments1 = input[:, :-stride].contiguous().view(batch_size, -1, frame_size)
    segments2 = input[:, stride:].contiguous().view(batch_size, -1, frame_size)
    segments = torch.cat([segments1, segments2], 2).view(batch_size, -1, frame_size)

    win = torch.sin(torch.Tensor(numpy.pi*numpy.array(range(stride))/(stride-1))/2)
    win = torch.cat([win, win.flip(0)], 0)
    #print((segments * win).shape)
    if pyv == '180':
        segments = torch.fft.rfft(segments * win, frame_size, 2)
        # print(segments.shape)
        sh = segments.shape
        segments = torch.cat(
            [torch.real(segments).view(sh[0], sh[1], sh[2], 1), torch.imag(segments).view(sh[0], sh[1], sh[2], 1)], 3)
    else:
        segments = torch.rfft(segments * win, 1)
    # segments = segments.view(batch_size, -1)
    #segments = torch.log10((torch.sqrt(segments[:, :, :, 0] ** 2 + segments[:, :, :, 1] ** 2)).view(batch_size, -1) + 1)
    #segments = torch.sqrt(segments[:, :, :, 0] ** 2 + segments[:, :, :, 1] ** 2).view(batch_size, -1)
    #segments = torch.sqrt(segments[:, :, :, 0] ** 2 + segments[:, :, :, 1] ** 2)
    segments = segments.view(batch_size, -1, rfft_size*2)
    segments = segments[:,:,:-2]

    return segments

def IMDCT_signal(mix_input, est_output, frame_size):
    #batch_size, nsample = mix_input.shape
    stride = frame_size // 2
    rfft_size = stride + 1
    FB_num = frame_size // 2

    # pad the signals at the end for matching the window/stride size
    '''
    rest = frame_size - (stride + nsample % frame_size) % frame_size
    if rest > 0:
        pad = torch.zeros(batch_size, rest).type(mix_input.type())
        mix_input = torch.cat([mix_input, pad], 1)
    pad_aux = torch.zeros(batch_size, stride).type(mix_input.type())
    mix_input = torch.cat([pad_aux, mix_input, pad_aux], 1)
    '''

    win = torch.sin(torch.Tensor(numpy.pi * numpy.array(range(stride)) / (stride - 1)) / 2).cuda()
    win = torch.cat([win, win.flip(0)], 0)
    
    B, C, T = est_output.shape
    
    s1_FFT_output = est_output.view(B, -1, FB_num, 2)
    pad_aux = torch.zeros(B, s1_FFT_output.shape[1], 1, 2).type(est_output.type())
    s1_FFT_output = torch.cat([s1_FFT_output, pad_aux], 2)
    if pyv == '180':
        s1_FFT_output = torch.complex(s1_FFT_output[:, :, :, 0], s1_FFT_output[:, :, :, 1])
        s1_output = torch.fft.irfft(s1_FFT_output, frame_size, 2) * win
    else:
        s1_output = torch.irfft(s1_FFT_output, 1, signal_sizes=[frame_size]) * win
    s1_output = (s1_output.view(-1, 2, stride)[1:, 0]+s1_output.view(-1, 2, stride)[:-1, 1]).view(B, 1, -1)

#    s2_FFT_output = est_output[:, 1].view(B, -1, FB_num, 2)
#    s2_FFT_output = torch.cat([s2_FFT_output, pad_aux], 2)
#    if pyv == '180':
#        s2_FFT_output = torch.complex(s2_FFT_output[:, :, :, 0], s2_FFT_output[:, :, :, 1])
#        s2_output = torch.fft.irfft(s2_FFT_output, frame_size, 2) * win
#    else:
#        s2_output = torch.irfft(s2_FFT_output, 1, signal_sizes=[frame_size]) * win
#    s2_output = (s2_output.view(-1, 2, stride)[1:, 0]+s2_output.view(-1, 2, stride)[:-1, 1]).view(B, 1, -1)
#
#    output = torch.cat([s1_output, s2_output], 1)

    return s1_output

def IMDCT_signal_one(mix_input, est_output, frame_size):
    #batch_size, nsample = mix_input.shape
    stride = frame_size // 2
    rfft_size = stride + 1

    # pad the signals at the end for matching the window/stride size
    '''
    rest = frame_size - (stride + nsample % frame_size) % frame_size
    if rest > 0:
        pad = torch.zeros(batch_size, rest).type(mix_input.type())
        mix_input = torch.cat([mix_input, pad], 1)
    pad_aux = torch.zeros(batch_size, stride).type(mix_input.type())
    mix_input = torch.cat([pad_aux, mix_input, pad_aux], 1)
    '''

    win = torch.sin(torch.Tensor(numpy.pi * numpy.array(range(stride)) / (stride - 1)) / 2).cuda()
    win = torch.cat([win, win.flip(0)], 0)
    s1_FFT_output = est_output[0, 0].view(1, -1, rfft_size, 2)
    if pyv == '180':
        s1_FFT_output = torch.complex(s1_FFT_output[:, :, :, 0], s1_FFT_output[:, :, :, 1])
        s1_output = torch.fft.irfft(s1_FFT_output, frame_size, 2) * win
    else:
        s1_output = torch.irfft(s1_FFT_output, 1, signal_sizes=[frame_size]) * win
    s1_output = (s1_output.view(-1, 2, stride)[1:, 0]+s1_output.view(-1, 2, stride)[:-1, 1]).view(1, -1)

    return s1_output

def IMDCT_signal_ori(mix_input, est_output, frame_size):
    batch_size, nsample = mix_input.shape
    stride = frame_size // 2
    rfft_size = stride + 1

    # pad the signals at the end for matching the window/stride size
    rest = frame_size - (stride + nsample % frame_size) % frame_size
    if rest > 0:
        pad = torch.zeros(batch_size, rest).type(mix_input.type())
        mix_input = torch.cat([mix_input, pad], 1)
    pad_aux = torch.zeros(batch_size, stride).type(mix_input.type())
    mix_input = torch.cat([pad_aux, mix_input, pad_aux], 1)

    segments1 = mix_input[:, :-stride].contiguous().view(batch_size, -1, frame_size)
    segments2 = mix_input[:, stride:].contiguous().view(batch_size, -1, frame_size)
    segments = torch.cat([segments1, segments2], 2).view(batch_size, -1, frame_size).cuda()

    win = torch.sin(torch.Tensor(numpy.pi*numpy.array(range(frame_size))/(frame_size-1))).cuda()
    #win = torch.ones(1,256).cuda()
    mix_input_segments = torch.rfft(segments * win, 1)
    # segments = segments.view(batch_size, -1)
    mix_input_SP = torch.sqrt(mix_input_segments[:, :, :, 0] ** 2 + mix_input_segments[:, :, :, 1] ** 2).view(batch_size, -1)
    IRM = (10**est_output-1)/(mix_input_SP+1e-10)
    #IRM = torch.ones(IRM.shape).cuda()
    #IRM = torch.max(torch.min(IRM, torch.ones(IRM.shape).cuda()), torch.zeros(IRM.shape).cuda())numpy
    s1_FFT_output = mix_input_segments*IRM[0, 0, :].view(-1, rfft_size, 1).repeat(1,1,2).view(1,-1,rfft_size,2)
    s1_output = torch.irfft(s1_FFT_output, 1, signal_sizes=[frame_size]) * win
    s1_output = (s1_output.view(-1, 2, stride)[1:, 0]+s1_output.view(-1, 2, stride)[:-1, 1]).view(1, -1)

    s2_FFT_output = mix_input_segments * IRM[0, 1, :].view(-1, rfft_size, 1).repeat(1, 1, 2).view(1, -1, rfft_size, 2)
    s2_output = torch.irfft(s2_FFT_output, 1, signal_sizes=[frame_size]) * win
    s2_output = (s2_output.view(-1, 2, stride)[1:, 0]+s2_output.view(-1, 2, stride)[:-1, 1]).view(1, -1)

    output = torch.cat([s1_output, s2_output], 0)

    return output


def overlap_and_add(signal, frame_step):
    """Reconstructs a signal from a framed representation.

    Adds potentially overlapping frames of a signal with shape
    `[..., frames, frame_length]`, offsetting subsequent frames by `frame_step`.
    The resulting tensor has shape `[..., output_size]` where

        output_size = (frames - 1) * frame_step + frame_length

    Args:
        signal: A [..., frames, frame_length] Tensor. All dimensions may be unknown, and rank must be at least 2.
        frame_step: An integer denoting overlap offsets. Must be less than or equal to frame_length.

    Returns:
        A Tensor with shape [..., output_size] containing the overlap-added frames of signal's inner-most two dimensions.
        output_size = (frames - 1) * frame_step + frame_length

    Based on https://github.com/tensorflow/tensorflow/blob/r1.12/tensorflow/contrib/signal/python/ops/reconstruction_ops.py
    """
    outer_dimensions = signal.size()[:-2]
    frames, frame_length = signal.size()[-2:]

    subframe_length = math.gcd(frame_length, frame_step)  # gcd=Greatest Common Divisor
    subframe_step = frame_step // subframe_length
    subframes_per_frame = frame_length // subframe_length
    output_size = frame_step * (frames - 1) + frame_length
    output_subframes = output_size // subframe_length

    # print(subframe_length)
    # print(signal.shape)
    # print(outer_dimensions)
    # subframe_signal = signal.view(*outer_dimensions, -1, subframe_length)
    subframe_signal = signal.reshape(*outer_dimensions, -1, subframe_length)

    frame = torch.arange(0, output_subframes).unfold(0, subframes_per_frame, subframe_step)
    frame = signal.new_tensor(frame).long()  # signal may in GPU or CPU
    frame = frame.contiguous().view(-1)

    result = signal.new_zeros(*outer_dimensions, output_subframes, subframe_length)
    result.index_add_(-2, frame, subframe_signal)
    result = result.view(*outer_dimensions, -1)
    return result

def remove_pad(inputs, inputs_lengths):
    """
    Args:
        inputs: torch.Tensor, [B, C, T] or [B, T], B is batch size
        inputs_lengths: torch.Tensor, [B]
    Returns:
        results: a list containing B items, each item is [C, T], T varies
    """
    results = []
    dim = inputs.dim()
    if dim == 3:
        C = inputs.size(1)
    for input, length in zip(inputs, inputs_lengths):
        if dim == 3: # [B, C, T]
            results.append(input[:,:length].view(C, -1).cpu().numpy())
        elif dim == 2:  # [B, T]
            results.append(input[:length].view(-1).cpu().numpy())
    return results
    
class angle(Function):
    """Similar to torch.angle but robustify the gradient for zero magnitude."""

    @staticmethod
    def forward(ctx, x: Tensor):
        ctx.save_for_backward(x)
        return torch.atan2(x.imag, x.real)

    @staticmethod
    def backward(ctx, grad: Tensor):
        (x,) = ctx.saved_tensors
        grad_inv = grad / (x.real.square() + x.imag.square()).clamp_min_(1e-10)
        return torch.view_as_complex(torch.stack((-x.imag * grad_inv, x.real * grad_inv), dim=-1))

def as_complex(x: Tensor):
    if torch.is_complex(x):
        return x
    if x.shape[-1] != 2:
        raise ValueError(f"Last dimension need to be of length 2 (re + im), but got {x.shape}")
    if x.stride(-1) != 1:
        x = x.contiguous()
    return torch.view_as_complex(x)

def power_compress(x):
    real = x[..., 0]
    imag = x[..., 1]
    spec = torch.complex(real, imag)
    mag = torch.abs(spec)
    phase = torch.angle(spec)
    mag = mag**0.3
    real_compress = mag * torch.cos(phase)
    imag_compress = mag * torch.sin(phase)
    return torch.stack([real_compress, imag_compress], -1)

def power_uncompress(real, imag):
    spec = torch.complex(real, imag)
    mag = torch.abs(spec)
    phase = torch.angle(spec)
    mag = mag**(1./0.3)
    real_compress = mag * torch.cos(phase)
    imag_compress = mag * torch.sin(phase)
    return torch.stack([real_compress, imag_compress], -1)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise Exception('Boolean value expected.')
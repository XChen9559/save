#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time     : 19-11-1 09:12:12
# @Author   : zm
# @File     : model.py
# @Software : PyCharm

import torch
import torch.nn as nn
from models import FaSNet_base
#import ipdb


class models_all(nn.Module):
    def __init__(self, enc_dim, feature_dim, hidden_dim, layer, group_size=34, segment_size=250, trunk_size=12,
                 nspk=2, win_len=2):
        super(models_all, self).__init__()
        self.model2 = FaSNet_base(enc_dim=enc_dim, feature_dim=feature_dim, hidden_dim=hidden_dim, layer=layer, group_size=group_size, segment_size=segment_size, trunk_size=trunk_size, nspk = nspk, win_len = win_len)
        self.model2.load_state_dict(torch.load('./exp_fitune_tvnoise/pretrianed.pth.tar'))
        self.model2.requires_grad=True
        self.mel_fc=nn.Linear(64,group_size)
        self.mel_conv = nn.Conv2d(1, enc_dim, kernel_size=1, stride=1) #,bias=False
        self.enc_dim = enc_dim
    def forward(self, input1,input2, h_list1):
        #print(input2.shape)
        mel_feat = self.mel_fc(input2).unsqueeze(dim=1)
        #print(mel_feat.shape)
        mel_feat = self.mel_conv(mel_feat)
        batch,T = mel_feat.shape[0],mel_feat.shape[2] 
        mel_feat = mel_feat.permute(0,2,1,3).reshape(batch*T,self.enc_dim,-1)
        est_source,h_list1 =self.model2(input1,mel_feat,h_list1)
        return est_source, h_list1
        
if __name__=='__main__':
    from thop import profile
    from thop import clever_format
    print("<<Test Flops Begin>>")
    #input.shape torch.Size([1, 252, 512])  4s?
    #embeddingVector.shape torch.Size([1, 128])
    batch=1
    x1 = torch.rand(batch, 8, 1024).cuda()
    x2 = torch.rand(batch, 8, 64).cuda()  #embeddingVector
    h_list = []
    for i_list in range(5):
        h = torch.zeros([1, batch*37,32]).type(x1.type())
        h_list.append(h)
    rfft_size = 512
    model = models_all(enc_dim=96, feature_dim=32, hidden_dim=32, layer=5, group_size=37, segment_size=1024//2, trunk_size=8, nspk = 1, win_len = 2).cuda()
    
    y, _ = model(x1, x2,h_list)
    print(y.shape)
    macs, params = profile(model, inputs=(x1,x2, h_list, ))
    macs, params = clever_format([macs, params], "%.3f")
    print(macs)
    print(params)
    print("<<Test Flops End>>")
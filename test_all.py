
import torch
import os
import numpy as np

from models_syn  import models_all
from utils import MDCT_signal_hop960,IMDCT_signal_hop960,mel_octal
import librosa
import soundfile as sf
frame_size=1536 #1024
rfft_size = frame_size//2
#file_path ='/home/sr5/ruijie.meng/data/TV_noise/HQ_2024_4/'
#file_list = os.listdir(file_path)
#out_path = './result/'
#num_file=0
#trunk_size=8
#for filename in file_list:
#    num_file+=1    
#    #data,_ = librosa.load(file_path+filename,48000)
#    #data_16k = librosa.resample(data,48000,16000)
#    data_48k,sr = sf.read(file_path+filename)
#    
#    
#    h1 = torch.zeros([5,1,37,32]).float().cuda()
#    h2 = torch.zeros([5,1,37,32]).float().cuda()
#    last_flag=False
#    model = models_all(enc_dim=96, feature_dim=32, hidden_dim=32, layer=5, group_size=37, segment_size=1024//2, trunk_size=trunk_size, nspk = 1, win_len = 2)
#   
#    #model = FaSNet_base(enc_dim=64, feature_dim=32, hidden_dim=32, layer=2, segment_size=rfft_size, nspk=2, win_len=1)
#    model_state = torch.load('./exp_fitune_tvnoise/epoch5.pth.tar')
#    model.load_state_dict(model_state, strict=False)
#    model.cuda()
#    model.eval()
#    with torch.no_grad():
#
#             inputs1_L = torch.from_numpy(data_48k).unsqueeze(dim=0)
#             #print(inputs1_L.shape)
#             fft_inputL = MDCT_signal_hop960(inputs1_L,frame_size).float().cuda() 
#             rest = fft_inputL.shape[1] - fft_inputL.shape[1]//trunk_size*trunk_size         
#             if rest >0:
#                fft_inputL = torch.cat([fft_inputL,torch.zeros(fft_inputL.shape[0],trunk_size-rest,fft_inputL.shape[2]).cuda()],dim=1)
#               
#              
#             fft_inputL1 = fft_inputL[:,:,:1024]
#              
#             temp = fft_inputL1.reshape(1,-1,512,2)
#             padded_mag = torch.abs(torch.complex(temp[:,:,:,0],temp[:,:,:,1]))
#             
#             mel_filter = mel_octal(fs=32000,fft_size=512,nbins=64).unsqueeze(dim=0).cuda()          
#             #print(padded_mag.shape,mel_filter.shape,self.batch_size)
#             mel_feat = torch.matmul(padded_mag,mel_filter) #torch.mm(padded_mag,mel_filter) ## [T,F] * [F,Nbins]-->[T,Nbins]
#             #print(mel_feat.shape)
#             mel_feat_norm = mel_feat/(mel_feat.sum(2).unsqueeze(dim=2).repeat(1,1,64)+1e-8) 
#             mel_feat_norm = -mel_feat_norm *torch.log(mel_feat_norm+1e-8)   
#             
#             out_tflite_L,h1 = model(fft_inputL1,mel_feat_norm,h1)
#             
#             #print(fft_inputL.shape,out_tflite_L.shape)
#             
#             
#             out_tflite_L_spk=torch.cat([out_tflite_L.reshape(1,-1,1024),fft_inputL[:,:,1024:]],-1)#np.zeros((1,12,1024))    
#             ####### overlap and add          
#             est_spk_L = IMDCT_signal_hop960(out_tflite_L_spk,frame_size) #frame_size
#             
#             est_spk_L = est_spk_L.squeeze().cpu().numpy()
#             print(est_spk_L.shape,data_48k.shape)
#         #sf.write(out_path +filename.split('.wav')[0]+'_torch_ep6_1.wav',stereo[:,:data_48k.shape[0]].T,48000,'PCM_32')#stereo[:,:data_48k.shape[0]].T,16000,'PCM_16') 
#             minilength = len(data_48k) if len(data_48k) < len(est_spk_L) else len(est_spk_L)
#             sf.write(out_path +filename.split('.wav')[0]+'_ep5_0401__2_spk.wav',est_spk_L[:minilength],48000,'PCM_16')#stereo[:,:data_48k.shape[0]].T,16000,'PCM_16')
#             sf.write(out_path +filename.split('.wav')[0]+'_ep5_0401__2_music.wav',data_48k[:minilength] -est_spk_L[:minilength],48000,'PCM_16')#stereo[:,:data_48k.shape[0]].T,16000,'PCM_16')      
             
             
###             
file_path ='/home/sr5/ruijie.meng/data/TV_noise/HQ_test_0221/'

file_list = os.listdir(file_path)
out_path = './result/'
num_file=0
trunk_size=8
for filename in file_list:
    num_file+=1    
    #data,_ = librosa.load(file_path+filename,48000)
    #data_16k = librosa.resample(data,48000,16000)
    data_48k,sr = sf.read(file_path+filename)
    data_48k = data_48k[:,0]
    
    h1 = torch.zeros([5,1,37,32]).float().cuda()
    h2 = torch.zeros([5,1,37,32]).float().cuda()
    last_flag=False
    model = models_all(enc_dim=96, feature_dim=32, hidden_dim=32, layer=5, group_size=37, segment_size=1024//2, trunk_size=trunk_size, nspk = 1, win_len = 2)
    
    #model = FaSNet_base(enc_dim=64, feature_dim=32, hidden_dim=32, layer=2, segment_size=rfft_size, nspk=2, win_len=1)
    model_state = torch.load('./exp_fitune_tvnoise/epoch9.pth.tar')
    model.load_state_dict(model_state, strict=False)
    model.cuda()
    model.eval()
    with torch.no_grad():

             inputs1_L = torch.from_numpy(data_48k).unsqueeze(dim=0)
             #print(inputs1_L.shape)
             fft_inputL = MDCT_signal_hop960(inputs1_L,frame_size).float().cuda() 
             rest = fft_inputL.shape[1] - fft_inputL.shape[1]//trunk_size*trunk_size         
             if rest >0:
                fft_inputL = torch.cat([fft_inputL,torch.zeros(fft_inputL.shape[0],trunk_size-rest,fft_inputL.shape[2]).cuda()],dim=1)
               
              
             fft_inputL1 = fft_inputL[:,:,:1024]
              
             temp = fft_inputL1.reshape(1,-1,512,2)
             padded_mag = torch.abs(torch.complex(temp[:,:,:,0],temp[:,:,:,1]))
             
             mel_filter = mel_octal(fs=32000,fft_size=512,nbins=64).unsqueeze(dim=0).cuda()          
             #print(padded_mag.shape,mel_filter.shape,self.batch_size)
             mel_feat = torch.matmul(padded_mag,mel_filter) #torch.mm(padded_mag,mel_filter) ## [T,F] * [F,Nbins]-->[T,Nbins]
             #print(mel_feat.shape)
             mel_feat_norm = mel_feat/(mel_feat.sum(2).unsqueeze(dim=2).repeat(1,1,64)+1e-8) 
             mel_feat_norm = -mel_feat_norm *torch.log(mel_feat_norm+1e-8)   
             
             out_tflite_L,h1 = model(fft_inputL1,mel_feat_norm,h1)
             
             #print(fft_inputL.shape,out_tflite_L.shape)
             
             out_tflite_L_spk=torch.cat([out_tflite_L.reshape(1,-1,1024),fft_inputL[:,:,1024:]],-1)#np.zeros((1,12,1024))    
             ####### overlap and add          
             est_spk_L = IMDCT_signal_hop960(out_tflite_L_spk,frame_size) #frame_size
             
             est_spk_L = est_spk_L.squeeze().cpu().numpy()
             print(est_spk_L.shape,data_48k.shape)
             minilength = len(data_48k) if len(data_48k) < len(est_spk_L) else len(est_spk_L)
         #sf.write(out_path +filename.split('.wav')[0]+'_torch_ep6_1.wav',stereo[:,:data_48k.shape[0]].T,48000,'PCM_32')#stereo[:,:data_48k.shape[0]].T,16000,'PCM_16') 
             sf.write(out_path +filename.split('.wav')[0]+'_ep9_spk.wav',est_spk_L[:minilength],48000,'PCM_16')#stereo[:,:data_48k.shape[0]].T,16000,'PCM_16')
             sf.write(out_path +filename.split('.wav')[0]+'_ep9_music.wav',data_48k[:minilength] - est_spk_L[:minilength],48000,'PCM_16')#stereo[:,:data_48k.shape[0]].T,16000,'PCM_16')      
             
             
         

import torch
import os
import numpy as np

from models  import FaSNet_base
from utils import MDCT_signal_hop960,IMDCT_signal_hop960
import librosa
import soundfile as sf
frame_size=1536 #1024
rfft_size = frame_size//2
file_path = '/home/sr5/ruijie.meng/data/TV_noise/test_mv_seq/' 
#

#'/home/sr5/ruijie.meng/voice_extraction/sound_case_erase/tf_code/TV_enhance/test_seq/1ch_16bit_32k/' #2ch_32bit
file_list = os.listdir(file_path)
out_path = './result/'
num_file=0
trunk_size=8
for filename in file_list:
    num_file+=1    
    #data,_ = librosa.load(file_path+filename,48000)
    #data_16k = librosa.resample(data,48000,16000)
    data_48k,sr = sf.read(file_path+filename)
    data_48k = data_48k[:,0]
    
    h1 = torch.zeros([5,1,37,32]).float().cuda()
    h2 = torch.zeros([5,1,37,32]).float().cuda()
    last_flag=False
    model = models_all(enc_dim=96, feature_dim=32, hidden_dim=32, layer=5, group_size=37, segment_size=1024//2, trunk_size=trunk_size, nspk = 1, win_len = 2)
    
    #model = FaSNet_base(enc_dim=64, feature_dim=32, hidden_dim=32, layer=2, segment_size=rfft_size, nspk=2, win_len=1)
    model_state = torch.load('./exp_fitune_tvnoise/epoch9.pth.tar')
    model.load_state_dict(model_state, strict=False)
    model.cuda()
    model.eval()
    with torch.no_grad():

             inputs1_L = torch.from_numpy(data_48k).unsqueeze(dim=0)
             #print(inputs1_L.shape)
             fft_inputL = MDCT_signal_hop960(inputs1_L,frame_size).float().cuda() 
             rest = fft_inputL.shape[1] - fft_inputL.shape[1]//trunk_size*trunk_size         
             if rest >0:
                fft_inputL = torch.cat([fft_inputL,torch.zeros(fft_inputL.shape[0],trunk_size-rest,fft_inputL.shape[2]).cuda()],dim=1)
               
              
             fft_inputL1 = fft_inputL[:,:,:1024]
              
             temp = fft_inputL1.reshape(1,-1,512,2)
             padded_mag = torch.abs(torch.complex(temp[:,:,:,0],temp[:,:,:,1]))
             
             mel_filter = mel_octal(fs=32000,fft_size=512,nbins=64).unsqueeze(dim=0).cuda()          
             #print(padded_mag.shape,mel_filter.shape,self.batch_size)
             mel_feat = torch.matmul(padded_mag,mel_filter) #torch.mm(padded_mag,mel_filter) ## [T,F] * [F,Nbins]-->[T,Nbins]
             #print(mel_feat.shape)
             mel_feat_norm = mel_feat/(mel_feat.sum(2).unsqueeze(dim=2).repeat(1,1,64)+1e-8) 
             mel_feat_norm = -mel_feat_norm *torch.log(mel_feat_norm+1e-8)   
             
             out_tflite_L,h1 = model(fft_inputL1,mel_feat_norm,h1)
             
             #print(fft_inputL.shape,out_tflite_L.shape)
             out_tflite_L_spk=torch.cat([out_tflite_L.reshape(1,-1,1024),fft_inputL[:,:,1024:]],-1)#np.zeros((1,12,1024))    
             ####### overlap and add          
             est_spk_L = IMDCT_signal_hop960(out_tflite_L_spk,frame_size) #frame_size
             
             est_spk_L = est_spk_L.squeeze().cpu().numpy()
             print(est_spk_L.shape,data_48k.shape)
             minilength = len(data_48k) if len(data_48k) < len(est_spk_L) else len(est_spk_L)
         #sf.write(out_path +filename.split('.wav')[0]+'_torch_ep6_1.wav',stereo[:,:data_48k.shape[0]].T,48000,'PCM_32')#stereo[:,:data_48k.shape[0]].T,16000,'PCM_16') 
             sf.write(out_path +filename.split('.wav')[0]+'_ep9_spk.wav',est_spk_L[:minilength],48000,'PCM_16')#stereo[:,:data_48k.shape[0]].T,16000,'PCM_16')
             sf.write(out_path +filename.split('.wav')[0]+'_ep9_music.wav',data_48k[:minilength] - est_spk_L[:minilength],48000,'PCM_16')#stereo[:,:data_48k.shape[0]].T,16000,'PCM_16')      
             
             
         
    
    
    
    
#    
#    
#    
#    
#    
#    
#    
#    
#    
#    
#    
    
    
    
    
    
    
        
#    
#    
#    
#    
#    
#    
#    
#    
    
    
    
    
    
    
    
    
    
    
    
    
    
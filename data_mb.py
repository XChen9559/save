# Created on 2018/12
# Author: Kaituo XU
"""
Logic:
1. AudioDataLoader generate a minibatch from AudioDataset, the size of this
   minibatch is AudioDataLoader's batchsize. For now, we always set
   AudioDataLoader's batchsize as 1. The real minibatch size we care about is
   set in AudioDataset's __init__(...). So actually, we generate the
   information of one minibatch in AudioDataset.
2. After AudioDataLoader getting one minibatch from AudioDataset,
   AudioDataLoader calls its collate_fn(batch) to process this minibatch.

Input:
    Mixtured WJS0 tr, cv and tt path
Output:
    One batch at a time.
    Each inputs's shape is B x T
    Each targets's shape is B x C x T
"""

import json
import math
import os

import numpy as np
import torch
import torch.utils.data as data
from audiolib import segmental_snr_mixer
import librosa
import copy
import random
class AudioDataset(data.Dataset):

    def __init__(self, json_dir, batch_size, sample_rate=8000, segment=4.0, cv_maxlen=8.0, flag_eval = False):
        """
        Args:
            json_dir: directory including mix.json, s1.json and s2.json
            segment: duration of audio segment, when set to -1, use full audio

        xxx_infos is a list and each item is a tuple (wav_file, #samples)
        """
        super(AudioDataset, self).__init__()
        noise_json = os.path.join(json_dir, 's1.json')
        voice_json = os.path.join(json_dir, 's1.json')
#        music_json = os.path.join(json_dir, 'music_shutter_48k.json')

        with open(voice_json, 'r') as f:
            voice_jsons = json.load(f)        
        with open(noise_json, 'r') as f:
            noise_jsons = json.load(f)
#        with open(music_json, 'r') as f:
#            music_jsons = json.load(f)
        
        # sort it by #samples (impl bucket)
        def sort(infos): return sorted(
            infos, key=lambda info: int(info[1]), reverse=True)
        sorted_voice_infos = sort(voice_jsons)
        sorted_noise_infos = sort(noise_jsons)
        sorted_music_infos = sort(noise_jsons)
        
        if flag_eval == True:
           sorted_voice_infos = sorted_voice_infos[:10000] ##cv 20220  #tr 181990
           sorted_noise_infos = sorted_noise_infos[:10000]
           sorted_music_infos = sorted_music_infos[:10000]
           sorted_music_infos_others = sorted_music_infos
           sorted_vocal_infos = sorted_music_infos[:10000]
        if flag_eval == False:
            ### instrument
            a_json = os.path.join('/common-data/mengruijie/data/wsj-2mix-16k/json/tr/s1.json')
            with open(a_json, 'r') as f:
                bass_infos = json.load(f)
            # a_json = os.path.join('/home/sr5/ruijie.meng/data/musicdb_48k_4s/json/train/drums.json')
            # with open(a_json, 'r') as f:
            #     drums_infos = json.load(f)
            # a_json = os.path.join('/home/sr5/ruijie.meng/data/musicdb_48k_4s/json/train/mixture_wovocal.json')
            # with open(a_json, 'r') as f:
            #     mixture_wovocal_infos = json.load(f)
            # a_json = os.path.join('/home/sr5/ruijie.meng/data/musicdb_48k_4s/json/train/other.json')
            # with open(a_json, 'r') as f:
            #     other_infos = json.load(f)
            # a_json = os.path.join('/home/sr5/ruijie.meng/data/music/music_48k.json')
            # with open(a_json, 'r') as f:
            #     music_infos = json.load(f)
            #
            # a_json = os.path.join('/home/sr5/ruijie.meng/data/TV_noise/48k/instrument.json')
            # with open(a_json, 'r') as f:
            #     music_mv_infos = json.load(f)
            #
            # a_json = os.path.join('/home/sr5/ruijie.meng/data/TV_noise/48k/20240202_electric_guitar_48k_4s.json')
            # with open(a_json, 'r') as f:
            #     music_guitar_infos1 = json.load(f)
            #
            # a_json = os.path.join('/home/sr5/ruijie.meng/data/TV_noise/48k/20240204_bilibili_electric_guitar_wav_48k_4s.json')
            # with open(a_json, 'r') as f:
            #     music_guitar_infos2 = json.load(f)
            #
            # a_json = os.path.join('/home/sr5/ruijie.meng/data/TV_noise/48k/20240223_bilibili_guitar_48k_4s.json')
            # with open(a_json, 'r') as f:
            #     music_guitar_infos3 = json.load(f)
            # #music_guitar_infos = music_guitar_infos1 + music_guitar_infos2 + music_guitar_infos3
            #
            #
            # a_json = os.path.join('/home/sr5/ruijie.meng/data/TV_noise/48k/20240221_bilibili_bass_48k_4s.json')
            # with open(a_json, 'r') as f:
            #     music_bass_infos = json.load(f)
            # a_json = os.path.join('/home/sr5/ruijie.meng/data/TV_noise/48k/20240221_bilibili_bass_drum_48k_4s.json')
            # with open(a_json, 'r') as f:
            #     music_drums_infos = json.load(f)
            #
            # a_json = os.path.join('/home/sr5/ruijie.meng/data/TV_noise/48k/instrument_0225/flute_split.json')
            # with open(a_json, 'r') as f:
            #     music_flute_infos = json.load(f)
            #
            # a_json = os.path.join('/home/sr5/ruijie.meng/data/TV_noise/48k/instrument_0225/sax_split.json')
            # with open(a_json, 'r') as f:
            #     music_sax_infos = json.load(f)
            #
            # a_json = os.path.join('/home/sr5/ruijie.meng/data/TV_noise/48k/instrument_0225/Symphony_split.json')
            # with open(a_json, 'r') as f:
            #     music_Symphony_infos = json.load(f)
            #
            # a_json = os.path.join('/home/sr5/ruijie.meng/data/TV_noise/48k/instrument_0225/trump_split.json')
            # with open(a_json, 'r') as f:
            #     music_trump_infos = json.load(f)
            #
            # a_json = os.path.join('/home/sr5/ruijie.meng/data/TV_noise/48k/instrument_0225/tuba_split.json')
            # with open(a_json, 'r') as f:
            #     music_tuba_infos = json.load(f)
            # a_json = os.path.join('/home/sr5/ruijie.meng/data/TV_noise/48k/20240301_mrj_48k_4s.json')
            # with open(a_json, 'r') as f:
            #     volins_infos = json.load(f)
            #
            # a_json = os.path.join('/home/sr5/ruijie.meng/data/separation3/slakh.json')
            # with open(a_json, 'r') as f:
            #     slakh_infos = json.load(f)
            #
            # sorted_music_infos = bass_infos + drums_infos + mixture_wovocal_infos + other_infos + music_infos +music_mv_infos+music_guitar_infos1+music_guitar_infos2+music_guitar_infos3\
            #                      +music_bass_infos+music_drums_infos+music_flute_infos+music_sax_infos+music_Symphony_infos+music_trump_infos+music_tuba_infos+volins_infos+slakh_infos
            #
               
            ### vocal
                a_json = os.path.join('/common-data/mengruijie/data/wsj-2mix-16k/json/tr/s1.json')
            with open(a_json, 'r') as f:
                 musicdb_vocal_indos = json.load(f)

            # a_json = os.path.join('/home/sr5/ruijie.meng/data/TV_noise/48k/songs_acapella_split.json')
            # with open(a_json, 'r') as f:
            #     songs_acapella_infos = json.load(f)
            #
            # a_json = os.path.join('/home/sr5/ruijie.meng/data/TV_noise/48k/songs_blibli.json')
            # with open(a_json, 'r') as f:
            #     songs_bili_infos = json.load(f)
            #
            # a_json = os.path.join('/home/sr5/ruijie.meng/data/chorus_vocal/EsmucChoirDataset_split.json')
            # with open(a_json, 'r') as f:
            #     EsmucChoirDataset_infos = json.load(f)
            #
            # a_json = os.path.join('/home/sr5/ruijie.meng/data/TV_noise/48k/0205_vocals_split.json')
            # with open(a_json, 'r') as f:
            #     songs_0205vocals_infos = json.load(f)
            # vocal_infos = musicdb_vocal_indos +   songs_acapella_infos +songs_bili_infos +EsmucChoirDataset_infos + songs_0205vocals_infos  #15347
            #
            
            ### noise
            a_json = os.path.join('/common-data/mengruijie/data/wsj-2mix-16k/json/tr/s1.json')
            with open(a_json, 'r') as f:
                dns_infos = json.load(f) ##28014
                
            # a_json = os.path.join('/home/sr5/adv_audio_lab_DNN/USER/DB_SRCB/DB/babble_48k_4s.json')
            # with open(a_json, 'r') as f:
            #     babble_infos = json.load(f)
            #
            # a_json = os.path.join('/home/sr5/adv_audio_lab_DNN/USER/DB_SRCB/DB/wind_48k.json')
            # with open(a_json, 'r') as f:
            #     wind_infos = json.load(f)
            #
            # a_json = os.path.join('/home/sr5/ruijie.meng/data/TV_noise/48k/effect_sound48k.json')
            # with open(a_json, 'r') as f:
            #     movie_effect_sound_infos = json.load(f)
            #
            # a_json = os.path.join('/home/sr5/ruijie.meng/data/TV_noise/48k/Movie_remove_human_voice_48k_4s.json')
            # with open(a_json, 'r') as f:
            #     movie_noise_infos1 = json.load(f)
            #
            # a_json = os.path.join('/home/sr5/ruijie.meng/data/TV_noise/48k/20240130_movie_remove_human_voice_48k_4s.json')
            # with open(a_json, 'r') as f:
            #     movie_noise_infos2 = json.load(f)
            # movie_noise_infos = movie_noise_infos1 + movie_noise_infos2
            #
            # a_json = os.path.join('/home/sr5/ruijie.meng/data/TV_noise/48k/movie_trailer_split.json')
            # with open(a_json, 'r') as f:
            #     movie_trailer_infos = json.load(f)
            #
            # a_json = os.path.join('/home/sr5/ruijie.meng/data/separation3/noise_split.json')
            # with open(a_json, 'r') as f:
            #     noise_others_infos = json.load(f) #38668
                      
#            ### noise 74611  ; voice : 181990;  15347
            #print(len(sorted_noise_infos),len(babble_infos + wind_infos +movie_effect_sound_infos +movie_noise_infos +movie_trailer_infos)) 
            #noise_mv =  babble_infos + wind_infos +movie_effect_sound_infos +movie_noise_infos +movie_trailer_infos  ##
            #print(len(noise_mv),len(sorted_noise_infos),len(noise_others_infos),len(vocal_infos))
            # sorted_noise_infos =  dns_infos+noise_others_infos + noise_mv  # 28014 + 38668 + 7929
            # sorted_noise_infos =  sorted_noise_infos + dns_infos+noise_others_infos + noise_mv
            # sorted_noise_infos =  sorted_noise_infos +  dns_infos +  dns_infos
            # sorted_noise_infos = sorted_noise_infos[0:len(sorted_voice_infos)]
            #
            # sorted_vocal_infos = vocal_infos+vocal_infos+vocal_infos+vocal_infos+vocal_infos+vocal_infos+vocal_infos+vocal_infos+vocal_infos+vocal_infos+vocal_infos+vocal_infos
            # sorted_vocal_infos = sorted_vocal_infos[0:len(sorted_voice_infos)]
            #
            # sorted_music_infos = sorted_music_infos + sorted_music_infos
            # sorted_music_infos = sorted_music_infos[0:len(sorted_voice_infos)]
            #
            # sorted_music_infos_others = copy.copy(sorted_music_infos)
            #
            # sorted_music_infos = sorted_music_infos[0:len(sorted_voice_infos)]

            sorted_music_infos_others = copy.copy(sorted_music_infos)
            np.random.shuffle(sorted_music_infos_others)
            sorted_vocal_infos = musicdb_vocal_indos
            sorted_music_infos = bass_infos
            sorted_noise_infos = dns_infos
                                       
            print(len(sorted_noise_infos),len(sorted_music_infos),len(sorted_voice_infos),len(sorted_vocal_infos))
        segment = -1    
        if segment >= 0.0:
            # segment length and count dropped utts
            segment_len = int(segment * sample_rate)  # 4s * 8000/s = 32000 samples
            drop_utt, drop_len = 0, 0
            for _, sample in sorted_voice_infos:
                if sample < segment_len:
                    drop_utt += 1
                    drop_len += sample
            #print('****************',segment_len,batch_size)
            print("Drop {} utts({:.2f} h) which is short than {} samples".format(
                drop_utt, drop_len/sample_rate/36000, segment_len))
            # generate minibach infomations
            minibatch = []
            start = 0
            while True:
                num_segments = 0
                end = start
                part_voice,part_noise, part_vocal,part_music,part_music_others = [], [], [],[],[]
                while num_segments < batch_size and end < len(sorted_voice_infos):
                    utt_len = int(sorted_voice_infos[end][1])
                    #print('00000000000000000000000000000000000',utt_len,segment_len)
                    if utt_len >= segment_len:  # skip too short utt
                        num_segments += math.ceil(utt_len / segment_len)
                        # Ensure num_segments is less than batch_size
                        if num_segments > batch_size:
                            # if num_segments of 1st audio > batch_size, skip it
                            if start == end:                                
                                part_voice.append(sorted_voice_infos[end])
                                part_noise.append(sorted_noise_infos[end])
                                part_vocal.append(sorted_vocal_infos[end])
                                part_music.append(sorted_music_infos[end])
                                part_music_others.append(sorted_music_infos_others[end])
                                end += 1
                            break
                        
                        part_voice.append(sorted_voice_infos[end])
                        part_noise.append(sorted_noise_infos[end])
                        part_vocal.append(sorted_vocal_infos[end])
                        part_music.append(sorted_music_infos[end])
                        part_music_others.append(sorted_music_infos_others[end])
                    end += 1
                if len(part_mix) > 0:
                    minibatch.append([part_voice, part_noise,part_vocal,part_music, part_music_others,
                                      sample_rate, segment_len,flag_eval])
                if end == len(sorted_s1_infos):
                    break
                start = end
            self.minibatch = minibatch
        else:  # Load full utterance but not segment
            # generate minibach infomations
            minibatch = []
            start = 0
            while True:
                end = min(len(sorted_voice_infos), start + batch_size)
                # Skip long audio to avoid out-of-memory issue
#                if int(sorted_mix_infos[start][1]) > cv_maxlen * sample_rate:
#                    start = end
#                    continue
                minibatch.append([
                                  sorted_voice_infos[start:end],
                                  sorted_noise_infos[start:end],
                                  sorted_vocal_infos[start:end],
                                  sorted_music_infos[start:end],
                                  sorted_music_infos_others[start:end],
                                  sample_rate, segment,flag_eval])
                if end == len(sorted_voice_infos):
                    break
                start = end
            self.minibatch = minibatch

    def __getitem__(self, index):
        return self.minibatch[index]

    def __len__(self):
        return len(self.minibatch)

class AudioDataLoader(data.DataLoader):
    """
    NOTE: just use batchsize=1 here, so drop_last=True makes no sense here.
    """

    def __init__(self, *args, **kwargs):
        super(AudioDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn

def _collate_fn(batch):
    """
    Args:
        batch: list, len(batch) = 1. See AudioDataset.__getitem__()
    Returns:
        mixtures_pad: B x T, torch.Tensor
        ilens : B, torch.Tentor
        sources_pad: B x C x T, torch.Tensor
    """
    # batch should be located in list
    assert len(batch) == 1
    #print('********************',batch[0])
    mixtures, sources = load_mixtures_and_sources(batch[0])
    #print('************',len(mixtures), len(sources))
    # get batch of lengths of input sequences
    ilens = np.array([mix.shape[0] for mix in mixtures])

    # perform padding and convert to tensor
    pad_value = 0
    mixtures_pad = pad_list([torch.from_numpy(mix).float()
                             for mix in mixtures], pad_value)
    ilens = torch.from_numpy(ilens)
    sources_pad = pad_list([torch.from_numpy(s).float()
                            for s in sources], pad_value)
    # N x T x C -> N x C x T
    sources_pad = sources_pad.permute((0, 2, 1)).contiguous()
    #print('mixtures_pad.shape {}'.format(mixtures_pad.shape),sources_pad.shape,mixtures[0].shape)
    #print('ilens {}'.format(ilens))
    return mixtures_pad, ilens, sources_pad #,filename

# Eval data part
from preprocess import preprocess_one_dir

class EvalDataset(data.Dataset):

    def __init__(self, mix_dir, mix_json, batch_size, sample_rate=8000):
        """
        Args:
            mix_dir: directory including mixture wav files
            mix_json: json file including mixture wav files
        """
        super(EvalDataset, self).__init__()
        assert mix_dir != None or mix_json != None
        if mix_dir is not None:
            # Generate mix.json given mix_dir
            preprocess_one_dir(mix_dir, mix_dir, 'mix',
                               sample_rate=sample_rate)
            mix_json = os.path.join(mix_dir, 'mix_single.json')
        with open(mix_json, 'r') as f:
            mix_infos = json.load(f)
        # sort it by #samples (impl bucket)
        def sort(infos): return sorted(
            infos, key=lambda info: int(info[1]), reverse=True)
        sorted_mix_infos = sort(mix_infos)
        # generate minibach infomations
        minibatch = []
        start = 0
        while True:
            end = min(len(sorted_mix_infos), start + batch_size)
            minibatch.append([sorted_mix_infos[start:end],
                              sample_rate])
            if end == len(sorted_mix_infos):
                break
            start = end
        self.minibatch = minibatch

    def __getitem__(self, index):
        return self.minibatch[index]

    def __len__(self):
        return len(self.minibatch)

class EvalDataLoader(data.DataLoader):
    """
    NOTE: just use batchsize=1 here, so drop_last=True makes no sense here.
    """

    def __init__(self, *args, **kwargs):
        super(EvalDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn_eval


def _collate_fn_eval(batch):
    """
    Args:
        batch: list, len(batch) = 1. See AudioDataset.__getitem__()
    Returns:
        mixtures_pad: B x T, torch.Tensor
        ilens : B, torch.Tentor
        filenames: a list contain B strings
    """
    # batch should be located in list
    assert len(batch) == 1
    mixtures, filenames = load_mixtures(batch[0])

    # get batch of lengths of input sequences
    ilens = np.array([mix.shape[0] for mix in mixtures])

    # perform padding and convert to tensor
    pad_value = 0
    mixtures_pad = pad_list([torch.from_numpy(mix).float()
                             for mix in mixtures], pad_value)
    ilens = torch.from_numpy(ilens)
    return mixtures_pad, ilens, filenames

# ------------------------------ utils ------------------------------------
def load_mixtures_and_sources(batch):
    """
    Each info include wav path and wav duration.
    Returns:
        mixtures: a list containing B items, each item is T np.ndarray
        sources: a list containing B items, each item is T x C np.ndarray
        T varies from item to item.
    """
    mixtures, sources ,filename= [], [],[]
    voice_infos, noise_infos,vocal_infos,music_infos, music_infos_others,sample_rate, segment_len,flag_eval = batch
    #print('infos',mix_infos)
    # for each utterance
    for voice_info, noise_info,vocal_info,music_info, music_info_others  in zip(voice_infos, noise_infos,vocal_infos,music_infos, music_infos_others):
        
        voice_path = voice_info[0]
        noise_path = noise_info[0]
        vocal_path = vocal_info[0]
        music_path = music_info[0]
        
        music_path_others = music_info_others[0]
        #print(mix_info[1],s1_info[1],s2_info[1])
        #assert mix_info[1] == s1_info[1] and s1_info[1] == s2_info[1]
        # read wav file
        voice, _ = librosa.load(voice_path, sr=sample_rate)
        noise, _ = librosa.load(noise_path, sr=sample_rate)
        vocal, _ = librosa.load(vocal_path, sr=sample_rate)
        music, _ = librosa.load(music_path, sr=sample_rate)
        music_others,_ = librosa.load(music_path_others, sr=sample_rate) ### noise
        
        pur_length = 4*48000
        if len(voice)>pur_length:
               random_index = np.random.randint(0,len(voice)-pur_length)
               voice = voice[random_index:random_index + pur_length]
        else:               
               voice = np.concatenate([voice,np.zeros((pur_length-len(voice)))],axis=0)#music = fix_length(music,len(s1))

        if len(vocal)>pur_length:
               random_index = np.random.randint(0,len(vocal)-pur_length)
               vocal = vocal[random_index:random_index + pur_length]
        else:               
               vocal = np.concatenate([vocal,np.zeros((pur_length-len(vocal)))],axis=0)#music = fix_length(music,len(s1))
                                      
        if len(music)>pur_length:
               random_index = np.random.randint(0,len(music)-pur_length)
               music = music[random_index:random_index + pur_length]
        else:               
               music = np.concatenate([music,np.zeros((pur_length-len(music)))],axis=0)#music = fix_length(music,len(s1))
            
        if len(music_others)>pur_length:
               random_index = np.random.randint(0,len(music_others)-pur_length)
               music_others = music_others[random_index:random_index + pur_length]
        else:               
               music_others = np.concatenate([music_others,np.zeros((pur_length-len(music_others)))],axis=0)#music = fix_length(music,len(s1))        

        if len(noise)>pur_length:
               random_index = np.random.randint(0,len(noise)-pur_length)
               noise = noise[random_index:random_index + pur_length]
        else:               
               noise = np.concatenate([noise,np.zeros((pur_length-len(noise)))],axis=0)#music = fix_length(music,len(s1))
                              
        if random.random() > 0.5: ##
            music = music+music_others           #random select one instrument and anothers
        
        music_flag =  random.random()  
        if  music_flag< 0.3: ##
            voice =voice+vocal 

        if  music_flag> 0.7: ##
            voice =vocal 
         
                                            
        snr = np.random.randint(-20,20)
        #s1, music, noisy = segmental_snr_mixer(clean=s1, noise = noise, snr=snr)
        
        case_flag = random.random()  ## case1: voice/vocal+noise  case2: voice/vocal+music  case3: voice/vocal+music+noise
        snr = np.random.randint(-20,20)
       
        if case_flag<0.3:
           noise_all = noise                   
           voice, noise_all, noisy = segmental_snr_mixer(clean=voice, noise = noise_all, snr=snr)
           s = np.dstack((voice, np.zeros_like(voice)))[0]  # T x C, C = 2
               
        if case_flag>0.7:
           noise_all = music
           voice, noise_all, noisy = segmental_snr_mixer(clean=voice, noise = noise_all, snr=snr)
           s = np.dstack((voice, noise_all))[0]  # T x C, C = 2
           
        if ((case_flag<=0.7) & (case_flag>=0.3)):
           snr1 = np.random.randint(-5,5)
           music, noise, noise_all = segmental_snr_mixer(clean=music, noise = noise, snr=snr1)
            
           E_noise = (np.mean(noise_all**(2)))**0.5
           E_voice = (np.mean(voice**(2)))**0.5
           if E_noise >1e-4:
              alpha =  E_voice/(E_noise+1e-8)/(10**(snr/20)) 
           else:
              alpha =1
         
           noise_all =  alpha*noise_all
           music = alpha*music
           noise = alpha*noise
           noisy =   noise_all+voice
           
           noisy_rms_level = np.random.randint(-35,-15)
           E_noisy = (np.mean(noisy**(2)))**0.5
           final_a = 10**(noisy_rms_level/20)/(E_noisy+1e-8)
           
           noisy = noisy*final_a
           voice = voice*final_a
           music = music*final_a
           noise = noise*final_a
           
           #voice, noise_all, noisy = segmental_snr_mixer(clean=voice, noise = noise_all, snr=snr)
           s = np.dstack((voice, music))[0]  # T x C, C = 2
   
        
        
        utt_len = noisy.shape[-1]

        segment_len = 4*48000
        if utt_len > segment_len:
          start_flag = np.random.randint(0,len(noisy)-segment_len)
          mixtures.append(noisy[start_flag:start_flag+segment_len])
          sources.append(s[start_flag:start_flag+segment_len,:])
        else:
          noisy = np.concatenate([noisy,np.zeros((segment_len-utt_len))],axis=0)
          s = np.concatenate([s,np.zeros((segment_len-utt_len,2))],axis=0)
          #print('bbbbbbbbbbbbbbbbbbbbb',noisy.shape,s.shape)
          mixtures.append(noisy)
          sources.append(s)
        #print('ssssssssssssssssssssssssssssssssss',noisy.shape,s.shape,mixtures[0].shape,sources[0].shape,len(mixtures))  
        filename.append(voice_path.split('./')[-1])
    return mixtures, sources

def load_mixtures(batch):
    """
    Returns:
        mixtures: a list containing B items, each item is T np.ndarray
        filenames: a list containing B strings
        T varies from item to item.
    """
    mixtures, filenames = [], []
    mix_infos, sample_rate = batch
    # for each utterance
    for mix_info in mix_infos:
        mix_path = mix_info[0]
        # read wav file
        mix, _ = librosa.load(mix_path, sr=sample_rate)
        mixtures.append(mix)
        filenames.append(mix_path)
    return mixtures, filenames

def pad_list(xs, pad_value):
    n_batch = len(xs)
    max_len = max(x.size(0) for x in xs)
    pad = xs[0].new(n_batch, max_len, * xs[0].size()[1:]).fill_(pad_value)
    for i in range(n_batch):
        pad[i, :xs[i].size(0)] = xs[i]
    return pad

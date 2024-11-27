import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
import numpy as np
import argparse
import librosa
import matplotlib.pyplot as plt
import torch

from utilities import create_folder, get_filename
from models import *
from pytorch_utils import move_data_to_device
import config


def audio_tagging(args):
    """Inference audio tagging result of an audio clip.
    """

    # Arugments & parameters
    sample_rate = args.sample_rate
    window_size = args.window_size
    hop_size = args.hop_size
    mel_bins = args.mel_bins
    fmin = args.fmin
    fmax = args.fmax
    model_type = args.model_type
    checkpoint_path = args.checkpoint_path
    audio_path = args.audio_path
    device = torch.device('cuda') if args.cuda and torch.cuda.is_available() else torch.device('cpu')
    
    classes_num = config.classes_num
    labels = config.labels

    # Model
    Model = eval(model_type)
    model = Model(sample_rate=sample_rate, window_size=window_size,
        hop_size=hop_size, mel_bins=mel_bins, fmin=fmin, fmax=fmax,
        classes_num=classes_num, train_from_scratch=True)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'], strict=False)

    # Parallel
    if 'cuda' in str(device):
        model.to(device)
        print('GPU number: {}'.format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)
    else:
        print('Using CPU.')
    
    # Load audio
    (waveform, _) = librosa.core.load(audio_path, sr=sample_rate, mono=True)
    audio_len = len(waveform)

    # loop and process 1s once
    clip_nums = audio_len//sample_rate
    label_list =[]
    for i in range(clip_nums):
        start = i * sample_rate
        end = start + sample_rate
        waveform_clip = waveform[None, start:end]
        waveform_clip = move_data_to_device(waveform_clip, device)

        # Forward
        with torch.no_grad():
            model.eval()
            batch_output_dict = model(waveform_clip, None)

        clipwise_output = batch_output_dict['clipwise_output'].data.cpu().numpy()[0]
        """(classes_num,)"""

        sorted_indexes = np.argsort(clipwise_output)[::-1]
        label_list.append(clipwise_output[0])
        # Print audio tagging top probabilities
        print('{}s~{}s:{}'.format(i, i+1, np.array(labels)[sorted_indexes[0]]))


    return label_list, labels


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Example of parser. ')
    subparsers = parser.add_subparsers(dest='mode')

    parser_at = subparsers.add_parser('audio_tagging')
    parser_at.add_argument('--sample_rate', type=int, default=16000)
    parser_at.add_argument('--window_size', type=int, default=512)
    parser_at.add_argument('--hop_size', type=int, default=160)
    parser_at.add_argument('--mel_bins', type=int, default=64)
    parser_at.add_argument('--fmin', type=int, default=50)
    parser_at.add_argument('--fmax', type=int, default=8000)
    parser_at.add_argument('--model_type', type=str, required=True)
    parser_at.add_argument('--checkpoint_path', type=str, required=True)
    parser_at.add_argument('--audio_path', type=str, required=True)
    parser_at.add_argument('--cuda', action='store_true', default=False)

    args = parser.parse_args()

    if args.mode == 'audio_tagging':
        audio_tagging(args)

    elif args.mode == 'sound_event_detection':
        sound_event_detection(args)

    else:
        raise Exception('Error argument!')
import os
import sys

sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
import numpy as np
import argparse
import time
import logging
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math

from thop import profile
from thop import clever_format

from config import (sample_rate, classes_num, mel_bins, fmin, fmax, window_size,
                    hop_size, window, pad_mode, center, ref, amin, top_db, clip_samples)
from losses import get_loss_func
from pytorch_utils import move_data_to_device, do_mixup, count_parameters, count_flops
from utilities import (create_folder, get_filename, create_logging, StatisticsContainer, Mixup, traverse_folder)
from data_generator import Dataset, TestSampler, collate_fn
from models import *
from evaluate import Evaluator


def eval_testset(args):
    # Arugments & parameters
    dataset_dir = args.dataset_dir
    workspace = args.workspace
    holdout_fold = args.holdout_fold
    batch_size = args.batch_size
    num_workers = 8
    sample_rate = args.sample_rate
    window_size = args.window_size
    hop_size = args.hop_size
    mel_bins = args.mel_bins
    fmin = args.fmin
    fmax = args.fmax
    model_type = args.model_type
    checkpoint_path = args.checkpoint_path
    device = torch.device('cuda') if args.cuda and torch.cuda.is_available() else torch.device('cpu')


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

    hdf5_path = os.path.join(workspace, 'features', 'testset_waveform.h5')
    dataset = Dataset()

    # Data generator
    testset_sampler = TestSampler(
        test_hdf5_path=hdf5_path,
        batch_size=batch_size)

    # Data loader
    test_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                  batch_sampler=testset_sampler, collate_fn=collate_fn,
                                                  num_workers=num_workers, pin_memory=True)


    # Evaluator
    evaluator = Evaluator(model=model)

    statistics = evaluator.evaluate(test_loader)
    print('Validate accuracy: {:.3f}, Validate F1 score: {:.3f}'.format(statistics['accuracy'],
                                                                               statistics['f1']))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Example of parser. ')
    subparsers = parser.add_subparsers(dest='mode')

    parser_at = subparsers.add_parser('eval_testset')
    parser_at.add_argument('--dataset_dir', type=str, required=True, help='Directory of dataset.')  # add params
    parser_at.add_argument('--workspace', type=str, required=True, help='Directory of your workspace.')
    parser_at.add_argument('--holdout_fold', type=str, choices=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'], required=True)
    parser_at.add_argument('--batch_size', type=int, required=True)
    parser_at.add_argument('--sample_rate', type=int, default=16000)
    parser_at.add_argument('--window_size', type=int, default=512)
    parser_at.add_argument('--hop_size', type=int, default=160)
    parser_at.add_argument('--mel_bins', type=int, default=64)
    parser_at.add_argument('--fmin', type=int, default=50)
    parser_at.add_argument('--fmax', type=int, default=8000)
    parser_at.add_argument('--model_type', type=str, required=True)
    parser_at.add_argument('--checkpoint_path', type=str, required=True)
    parser_at.add_argument('--cuda', action='store_true', default=False)

    args = parser.parse_args()

    if args.mode == 'eval_testset':
        eval_testset(args)

    elif args.mode == 'sound_event_detection':
        sound_event_detection(args)

    else:
        raise Exception('Error argument!')
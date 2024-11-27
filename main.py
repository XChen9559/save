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
from data_generator import Dataset, TrainSampler, EvaluateSampler, collate_fn
from models import Transfer_Cnn14, Cnn14, MobileNetV2, InvertedResidual
from evaluate import Evaluator



def train(args):

    # Arugments & parameters
    dataset_dir = args.dataset_dir
    workspace = args.workspace
    holdout_fold = args.holdout_fold
    model_type = args.model_type
    pretrained_checkpoint_path = args.pretrained_checkpoint_path
    freeze_base = args.freeze_base
    loss_type = args.loss_type
    augmentation = args.augmentation
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    resume_iteration = args.resume_iteration
    stop_iteration = args.stop_iteration
    device = 'cuda' if (args.cuda and torch.cuda.is_available()) else 'cpu'
    filename = args.filename
    num_workers = 8
    layer_num = args.layer_num
    train_from_scratch = args.train_from_scratch

    loss_func = get_loss_func(loss_type)
    if pretrained_checkpoint_path:
        pretrain = True
    else:
        if train_from_scratch:
            pretrain = False
        else:
            print("Conflict! Check setting train_from_scratch")
    
    hdf5_path = os.path.join(workspace, 'features', 'trainset_waveform.h5')

    checkpoints_dir = os.path.join(workspace, 'checkpoints', filename, 
        'holdout_fold={}'.format(holdout_fold), model_type, 'pretrain={}'.format(pretrain),
        'loss_type={}'.format(loss_type), 'augmentation={}'.format(augmentation),
         'batch_size={}'.format(batch_size), 'freeze_base={}_{}'.format(freeze_base,layer_num))
    create_folder(checkpoints_dir)

    statistics_path = os.path.join(workspace, 'statistics', filename, 
        'holdout_fold={}'.format(holdout_fold), model_type, 'pretrain={}'.format(pretrain),
        'loss_type={}'.format(loss_type), 'augmentation={}'.format(augmentation), 
        'batch_size={}'.format(batch_size), 'freeze_base={}_{}'.format(freeze_base,layer_num),
        'statistics.pickle')
    create_folder(os.path.dirname(statistics_path))
    
    logs_dir = os.path.join(workspace, 'logs', filename, 
        'holdout_fold={}'.format(holdout_fold), model_type, 'pretrain={}'.format(pretrain),
        'loss_type={}'.format(loss_type), 'augmentation={}'.format(augmentation), 
        'batch_size={}'.format(batch_size), 'freeze_base={}_{}'.format(freeze_base,layer_num))
    create_logging(logs_dir, 'w')
    logging.info(args)

    if 'cuda' in device:
        logging.info('Using GPU.')
    else:
        logging.info('Using CPU. Set --cuda flag to use GPU.')
    
    # Model
    Model = eval(model_type)                                                                                            # compile the string
    if train_from_scratch:
        model = Model(sample_rate, window_size, hop_size, mel_bins, fmin, fmax,
                      classes_num,train_from_scratch)
    else:
        model = Model(sample_rate, window_size, hop_size, mel_bins, fmin, fmax,
        classes_num, train_from_scratch)

    # add count params and flops
    # params_num = count_parameters(model)
    # flops_num = count_flops(model, clip_samples)
    # logging.info('Parameters num: {}'.format(params_num))
    # logging.info('Flops num: {:.3f} G'.format(flops_num / 1e9))

    # flops using THOP
    x1 = torch.rand(1, clip_samples)
    macs_num, params_num = profile(model, inputs=(x1,))
    logging.info('Parameters num: {:.3f} M'.format(params_num / 1e6))
    logging.info('macs num: {:.3f} M'.format(macs_num / 1e6))


    # Statistics
    statistics_container = StatisticsContainer(statistics_path)

    if pretrain:
        logging.info('Load pretrained model from {}'.format(pretrained_checkpoint_path))
        model.load_from_pretrain(pretrained_checkpoint_path)

    if resume_iteration:
        resume_checkpoint_path = os.path.join(checkpoints_dir, '{}_iterations.pth'.format(resume_iteration))
        logging.info('Load resume model from {}'.format(resume_checkpoint_path))
        resume_checkpoint = torch.load(resume_checkpoint_path)
        model.load_state_dict(resume_checkpoint['model'])
        statistics_container.load_state_dict(resume_iteration)
        iteration = resume_checkpoint['iteration']
    else:
        iteration = 0

    # Parallel
    print('GPU number: {}'.format(torch.cuda.device_count()))                                                           # print information
    model = torch.nn.DataParallel(model)                                # for multiple GPUs

    dataset = Dataset()

    # Data generator
    train_sampler = TrainSampler(
        hdf5_path=hdf5_path, 
        holdout_fold=holdout_fold, 
        batch_size=batch_size * 2 if 'mixup' in augmentation else batch_size)

    validate_sampler = EvaluateSampler(
        hdf5_path=hdf5_path, 
        holdout_fold=holdout_fold, 
        batch_size=batch_size)

    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=dataset, 
        batch_sampler=train_sampler, collate_fn=collate_fn, 
        num_workers=num_workers, pin_memory=True)                   # shuffle (default: False)

    validate_loader = torch.utils.data.DataLoader(dataset=dataset, 
        batch_sampler=validate_sampler, collate_fn=collate_fn, 
        num_workers=num_workers, pin_memory=True)

    if 'cuda' in device:
        model.to(device)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999),
        eps=1e-08, weight_decay=0., amsgrad=True)

    
    if 'mixup' in augmentation:
        mixup_augmenter = Mixup(mixup_alpha=1.)
     
    # Evaluator
    evaluator = Evaluator(model=model)
    
    train_bgn_time = time.time()
    
    # Train on batches
    for batch_data_dict in train_loader:

        # import crash
        # asdf
        
        # Evaluate
        if iteration % 200 == 0 and iteration > 0:
            if resume_iteration > 0 and iteration == resume_iteration:
                pass
            else:
                logging.info('------------------------------------')
                logging.info('Iteration: {}'.format(iteration))

                train_fin_time = time.time()

                statistics = evaluator.evaluate(validate_loader)
                logging.info('Validate accuracy: {:.3f}, Validate F1 score: {:.3f}'.format(statistics['accuracy'], statistics['f1']))

                statistics_container.append(iteration, statistics, 'validate')
                statistics_container.dump()

                train_time = train_fin_time - train_bgn_time
                validate_time = time.time() - train_fin_time

                logging.info(
                    'Train time: {:.3f} s, validate time: {:.3f} s'
                    ''.format(train_time, validate_time))

                train_bgn_time = time.time()

        # Save model 
        if iteration % 2000 == 0 and iteration > 0:
            checkpoint = {
                'iteration': iteration, 
                'model': model.module.state_dict()}

            checkpoint_path = os.path.join(
                checkpoints_dir, '{}_iterations.pth'.format(iteration))
                
            torch.save(checkpoint, checkpoint_path)
            logging.info('Model saved to {}'.format(checkpoint_path))
        
        if 'mixup' in augmentation:
            batch_data_dict['mixup_lambda'] = mixup_augmenter.get_lambda(len(batch_data_dict['waveform']))
        
        # Move data to GPU
        for key in batch_data_dict.keys():
            batch_data_dict[key] = move_data_to_device(batch_data_dict[key], device)
        
        # Train
        model.train()                                            #??? enabling batch normalization and dropout

        if 'mixup' in augmentation:
            batch_output_dict = model(batch_data_dict['waveform'], 
                batch_data_dict['mixup_lambda'])
            """{'clipwise_output': (batch_size, classes_num), ...}"""

            batch_target_dict = {'target': do_mixup(batch_data_dict['target'], 
                batch_data_dict['mixup_lambda'])}
            """{'target': (batch_size, classes_num)}"""
        else:
            batch_output_dict = model(batch_data_dict['waveform'], None)
            """{'clipwise_output': (batch_size, classes_num), ...}"""

            batch_target_dict = {'target': batch_data_dict['target']}
            """{'target': (batch_size, classes_num)}"""

        # loss and print
        loss = loss_func(batch_output_dict, batch_target_dict)

        audios_dir = os.path.join(dataset_dir)
        (audio_names, audio_paths) = traverse_folder(audios_dir)
        epoch_idx = math.floor((iteration+1)*batch_size/len(audio_names)*0.9)+1
        print('Epoch {0} | iter {1} | loss {2:.3f}| '.format(epoch_idx, iteration, loss))

        # Backward
        optimizer.zero_grad()                                #model params gradient = 0
        loss.backward()                                      #backward
        optimizer.step()                                     #update params

        # Stop learning
        if iteration == stop_iteration:
            break 

        iteration += 1
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Example of parser. ')                                                 # creatt argparser
    subparsers = parser.add_subparsers(dest='mode')

    # Train
    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--dataset_dir', type=str, required=True, help='Directory of dataset.')                   # add params
    parser_train.add_argument('--workspace', type=str, required=True, help='Directory of your workspace.')
    parser_train.add_argument('--holdout_fold', type=str, choices=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'], required=True)
    parser_train.add_argument('--model_type', type=str, required=True)
    parser_train.add_argument('--pretrained_checkpoint_path', type=str)
    parser_train.add_argument('--freeze_base', action='store_true', default=False)
    parser_train.add_argument('--loss_type', type=str, default='clip_bce', choices=['clip_bce'])
    parser_train.add_argument('--augmentation', type=str, choices=['none', 'mixup'], required=True)
    parser_train.add_argument('--learning_rate', type=float, required=True)
    parser_train.add_argument('--batch_size', type=int, required=True)
    parser_train.add_argument('--resume_iteration', type=int)
    parser_train.add_argument('--stop_iteration', type=int, required=True)
    parser_train.add_argument('--cuda', action='store_true', default=False)
    parser_train.add_argument('--layer_num', type=int, default=1)
    parser_train.add_argument('--train_from_scratch', action='store_true', default=False)                               # action how to process 'store-true' if param
                                                                                                                        # exists, store as True


    # Parse arguments
    args = parser.parse_args()                                                                                          # compile
    args.filename = get_filename(__file__)

    if args.mode == 'train':
        train(args)

    else:
        raise Exception('Error argument!')
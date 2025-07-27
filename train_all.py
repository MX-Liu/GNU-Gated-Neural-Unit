from future.moves import sys
from torch.multiprocessing import Pool
import subprocess
import shlex
import os
import argparse
import torch
import random
import numpy as np


def run_command(command):
    #print (command)
    process = subprocess.Popen(shlex.split(command), shell=False, stdout=subprocess.PIPE)
    while True:
        output = process.stdout.readline()
        if process.poll() is not None:
            break
        if output:
            print (output.strip())
    rc = process.poll()
    return rc

def hp_tuning(hp):

    # TODO REPLACE BY YOUR HYPERPARAMETERS

    command = ['python main.py',
                '--model', hp['model'],
                '--dataset', hp['dataset'],
                '--num_epochs', str(hp['num_epochs']),
                '--seed', str(hp['seed']),
                '--classifier', hp['classifier'],
                '--alpha1', str(hp['alpha1']),
                '--alpha2', str(hp['alpha2']),
                '--activation', hp['activation'],
                '--ds_percentage', str(hp['ds_percentage']),
                '--grid', str(hp['grid']),
                '--degree', str(hp['degree']),
                '--noise', str(hp['noise'])
                ]
    

    command = ' '.join(command)
    run_command(command)

    # log_path = os.path.join('logs', str(hp_sample) + '.json')
    # rs.write_params(log_path)


if __name__ == "__main__":

    
    parser = argparse.ArgumentParser(description="PyTorch Classification Template")
    
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['iris', 'mnist', 'cifar10'])
    parser.add_argument('--num_epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--classifier', type=str, default='mlp', choices=['linear', 'mlp', 'kan', 'gnu'], help='Classifier type for MLP-KAN.')
    parser.add_argument('--alpha1', type=float, default=0.5, help='Alpha value for MLP-KAN classifier.')
    parser.add_argument('--alpha2', type=float, default=0.5, help='Alpha value for MLP-KAN classifier.')
    parser.add_argument('--activation', type=str, default='relu', choices=['relu', 'silu','identity'], help='Activation function for MLP-KAN classifier.')
    parser.add_argument('--ds_percentage', type=float, default=1.0, help='Percentage of dataset to use for training (0.0 to 1.0).')
    parser.add_argument('--grid', type=int, default=5, help='Grid size for MLP-KAN classifier.')
    parser.add_argument('--degree', type=int, default=3, help='Degree for polynomial features in MLP-KAN classifier.')
    parser.add_argument('--noise', type=float, default=0.0, help='Noise level for datasets.')
    args = parser.parse_args()
    
    
    hp = []

    # data_name_list = ["dg", "uschad", "pamap2", "rw", "skodar", "dsads", "hapt", "oppo", "wisdm"]
    model_list = ['resnet18','resnet34', 'resnet50', 'vgg11', 'vgg16', 'vgg19','densenet121', 'densenet169', 'densenet201', 'densenet161', 'mobilenet_v2', 'googlenet']

    
    JOBS_PER_GPU = len(model_list)
    
    for i in range(JOBS_PER_GPU):
        hp_dic1 = {
            'model': model_list[i],
            "seed": args.seed,
            "dataset": args.dataset,
            "num_epochs": args.num_epochs,
            "classifier": args.classifier,
            "alpha1": args.alpha1,
            "alpha2": args.alpha2,
            "activation": args.activation,
            "ds_percentage": args.ds_percentage,
            "grid": args.grid,
            "degree": args.degree,
            "noise": args.noise,
        }
        hp.append(hp_dic1)
            
    pool = Pool(processes=JOBS_PER_GPU)
    pool.map(hp_tuning, hp, chunksize=1)
    pool.close()
    pool.join()


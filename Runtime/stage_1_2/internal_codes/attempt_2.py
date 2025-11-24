#!/usr/bin/env python3
# coding: utf-8

import sys
import os
from sklearn.preprocessing import binarize
import numpy as np
import pandas as pd
import math
import itertools
import glob
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
from torch import optim
from sklearn.preprocessing import binarize
import time

sys.path.append('/path/to/CDT_project/python_package')

from EinsumNetwork import Graph, EinsumNetwork
import wandb
import glob
import gzip
import tqdm
import pickle
import sparse

#############################################

def convert_bytes_to_readable(bytes):
    """
    Convert bytes to a human-readable format (MB or GB).
    """
    if bytes < 1024:
        return f"{bytes} B"
    elif bytes < 1024 * 1024:
        return f"{bytes / 1024:.2f} KB"
    elif bytes < 1024 * 1024 * 1024:
        return f"{bytes / (1024 * 1024):.2f} MB"
    else:
        return f"{bytes / (1024 * 1024 * 1024):.2f} GB"

device = torch.device("cuda") 

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if 'cuda' in device.type:
    torch.cuda.manual_seed(42)

def load_model_eval(
    sequence_length=1000,
    device='cuda',
):


    depth = 1
    num_repetitions = 22
    num_input_distributions = 22
    num_sums = 21
    
    
    max_num_epochs = 1000
    batch_size = 88
    
    early_stop_thresh = 5
    
    online_em_stepsize =  0.00018171143863824108
    online_em_frequency = 1
    
    expo_family = EinsumNetwork.CategoricalArray
    expo_family_args={'K': 2}

    graph = Graph.random_binary_trees(num_var=sequence_length, depth=depth, num_repetitions=num_repetitions)


    args = EinsumNetwork.Args(
        num_classes=1,
        num_input_distributions=num_input_distributions,
        exponential_family=expo_family,
        exponential_family_args=expo_family_args,
        num_sums=num_sums,
        num_var=sequence_length,
        online_em_frequency=online_em_frequency,
        online_em_stepsize=online_em_stepsize)
    
    einet = EinsumNetwork.EinsumNetwork(graph, args)
    einet.initialize()
    einet.to(device)

    dir_best_model = '/path/to/CDT_project/real_training/EM/best_models/best_model.pt'
    res_dict = torch.load(dir_best_model)

    einet.load_state_dict(res_dict['model_state_dict'])
    einet.eval()
    einet.compile()

    return einet

einet = load_model_eval()

print('gpu memory')
print(convert_bytes_to_readable(torch.cuda.mem_get_info()[0]),convert_bytes_to_readable(torch.cuda.mem_get_info()[1]))

##########################################################

numbers = list(range(1000))

# data_dir='combo_iterator'

LOG_EVERY_N = 1000000

def model_eval(array_zeros, combo):
    
    tensor = torch.from_numpy(array_zeros).to(torch.device(device))
    
    ll_sample = EinsumNetwork.eval_loglikelihood_batched(einet, tensor)  ### batch size

    data = ll_sample.detach().cpu().numpy()[:,0].astype(np.float16) ### converting to float16
    
    matrix = sparse.COO(combo.T, data, shape=(1000,1000,1000))

    return matrix


iterator = itertools.combinations(numbers,3)

def make_combo_quad(chunk_num, num_times, iterator, maximum_dim = 5000000):
    
    results=np.zeros([maximum_dim,3],dtype=np.int16)

    array_zeros = np.zeros((maximum_dim, 1000),dtype=np.int8)


    index =0
    for index, temp_combo in enumerate(iterator):
        # process the object from the iterator

        if (index % LOG_EVERY_N) == 0:
            print(f'iteration = {index}')

        results[index,0] = temp_combo[0]
        results[index,1] = temp_combo[1]
        results[index,2] = temp_combo[2]

        array_zeros[index,temp_combo] = 1

        index+=1

        
        if index == maximum_dim:
            matrix = model_eval(array_zeros, results)
            break

    if index < maximum_dim:

        array_zeros = array_zeros[:index,:]

        results = results[:index,:]

        matrix = model_eval(array_zeros, results)

        return False, chunk_num+1, matrix


    if chunk_num == num_times:
        print('*********************')
        print(f'chunk_num = {chunk_num}')
        return False, chunk_num+1, matrix
    else:
        print('*********************')
        print(f'chunk_num = {chunk_num}')
        return True, chunk_num+1, matrix
        
################################################

print('looping right now')
repeat = True

chunk_num = 0
num_times=-10
combined_matrix='string'


while repeat:
    repeat, chunk_num,matrix = make_combo_quad(chunk_num = chunk_num, num_times=num_times,iterator=iterator)

    if type(combined_matrix) == str:
        combined_matrix = matrix
    else:
        combined_matrix = sparse.elemwise(np.add, combined_matrix, matrix)

print('matrix looks like this')
print(combined_matrix)

print('saving right now')
sparse.save_npz("triple_joint_sparse_compile.npz", combined_matrix)

print('python all done')



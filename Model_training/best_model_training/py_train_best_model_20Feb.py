#!/usr/bin/env python3

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

#############################################

device = torch.device("cuda") 

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if 'cuda' in device.type:
    torch.cuda.manual_seed(42)

##############################################

def prepare_dataset():

	data_dir = '/path/to/CDT_project/train_test_leonie'

	counts_1 = pd.read_csv(f'{data_dir}/train.csv',index_col=0)

	part_20 = counts_1.sample(frac = 0.2)

	rest_part_80 = counts_1.drop(part_20.index)

	binary_array = binarize(rest_part_80, threshold=0)
	binary_20 = binarize(part_20,threshold=0)
	
	train_x = torch.from_numpy(binary_array).to(torch.device(device))
	val_x = torch.from_numpy(binary_20).to(torch.device(device))
	
	train_N, num_dims = train_x.shape
	val_N, _ = val_x.shape

	return train_x, train_N, num_dims, val_x, val_N

#################################################

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

#####################################################

wandb.init(entity='jy_learn', project="EM", name='train_best_model_20Feb')



train_x, train_N, num_dims, val_x, val_N = prepare_dataset()

graph = Graph.random_binary_trees(num_var=num_dims, depth=depth, num_repetitions=num_repetitions)


args = EinsumNetwork.Args(
    num_classes=1,
    num_input_distributions=num_input_distributions,
    exponential_family=expo_family,
    exponential_family_args=expo_family_args,
    num_sums=num_sums,
    num_var=num_dims,
    online_em_frequency=online_em_frequency,
    online_em_stepsize=online_em_stepsize)

einet = EinsumNetwork.EinsumNetwork(graph, args)
einet.initialize()
einet.to(device)


best_LL = 10000
best_epoch = -1

for epoch_count in range(max_num_epochs):

    # evaluate
    train_ll = EinsumNetwork.eval_loglikelihood_batched(einet, train_x)

    epoch_average_LL = -(train_ll / train_N)

    print(f'epoch: {epoch_count}, average LL: {epoch_average_LL}')
    # print("[{}]   train LL {} ".format(epoch_count,train_ll / train_N))
    #####################################################################
    

    val_ll = EinsumNetwork.eval_loglikelihood_batched(einet, val_x)

    epoch_val_LL = -(val_ll / val_N)

    wandb.log({"epoch": epoch_count, "NLL": epoch_average_LL, "val_NLL":epoch_val_LL})  ### minimise this thing

    # train
    idx_batches = torch.randperm(train_N).split(batch_size)
    for batch_count, idx in enumerate(idx_batches):
        batch_x = train_x[idx, :]
        outputs = einet.forward(batch_x)

        ll_sample = EinsumNetwork.log_likelihoods(outputs)
        log_likelihood = ll_sample.sum()

        objective = log_likelihood
        objective.backward()

        einet.em_process_batch()

    einet.em_update()

    if epoch_val_LL < best_LL:    ## changing this to validation
        best_LL = epoch_val_LL         ## changing this to validation
        best_epoch = epoch_count
        torch.save({'epoch': epoch_count, 'model_state_dict': einet.state_dict(),'val_NLL': epoch_val_LL, 'NLL': epoch_average_LL}, 'best_models/best_model_20Feb.pt')
  
        
    elif epoch_count - best_epoch > config.early_stop_thresh:
        break





print('python done')

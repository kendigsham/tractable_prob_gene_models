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

sweep_config = {'method': 'bayes', 'run_cap': 700,
'metric': {'goal': 'minimize', 'name': 'val_NLL'},
'parameters':{'max_num_epochs':{'value':500},
                              'depth':{'distribution': 'int_uniform','min':1, 'max':5},
                             'num_repetitions':{ 'distribution': 'int_uniform','min':1, 'max':50},
                             'num_input_distributions':{'distribution': 'int_uniform','min':1, 'max':30},
                             'num_sums':{'distribution': 'int_uniform','min':1, 'max':30},
                              'num_classes':{'value':1},
                              'batch_size':{'distribution': 'q_log_uniform_values','q': 8,'min': 32,'max': 256},
                              'early_stop_thresh':{'value':5},
                              'online_em_frequency': {'value': 1},
                              'online_em_stepsize' : {'distribution':'uniform', 'min': 0.000001, 'max':0.15}
                             }
               }



expo_family = EinsumNetwork.CategoricalArray
expo_family_args={'K': 2}

#####################################################

def train(config=None):
    with wandb.init(config=config):
        config = wandb.config

        train_x, train_N, num_dims, val_x, val_N = prepare_dataset()
        
        graph = Graph.random_binary_trees(num_var=num_dims, depth=config.depth, num_repetitions=config.num_repetitions)
        
        expo_family = EinsumNetwork.CategoricalArray
        
        args = EinsumNetwork.Args(
            num_classes=1,
            num_input_distributions=config.num_input_distributions,
            exponential_family=expo_family,
            exponential_family_args=expo_family_args,
            num_sums=config.num_sums,
            num_var=num_dims,
            online_em_frequency=config.online_em_frequency,
            online_em_stepsize=config.online_em_stepsize)
        
        einet = EinsumNetwork.EinsumNetwork(graph, args)
        einet.initialize()
        einet.to(device)
        
        
        best_LL = 10000
        best_epoch = -1
        
        for epoch_count in range(config.max_num_epochs):
        
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
            idx_batches = torch.randperm(train_N).split(config.batch_size)
            for batch_count, idx in enumerate(idx_batches):
                batch_x = train_x[idx, :]
                outputs = einet.forward(batch_x)
        
                ll_sample = EinsumNetwork.log_likelihoods(outputs)
                log_likelihood = ll_sample.sum()
        
                objective = log_likelihood
                objective.backward()
        
                einet.em_process_batch()
                # wandb.log({"batch_log_likelihood": log_likelihood}) ## this is slowing the training down
        
            einet.em_update()

            if epoch_average_LL < best_LL:
                best_LL = epoch_average_LL
                best_epoch = epoch_count
            elif epoch_count - best_epoch > config.early_stop_thresh:
                break



sweep_id = wandb.sweep(sweep_config, entity= "jy_learn" ,project="EM_14Feb")

wandb.agent(sweep_id, function=train)


print('python done')

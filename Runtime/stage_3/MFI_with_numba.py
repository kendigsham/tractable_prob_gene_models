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

sys.path.append('/home/ycks3/CDT_project/python_package')

from EinsumNetwork import Graph, EinsumNetwork
import wandb


#############################################

device = torch.device("cuda") 

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if 'cuda' in device.type:
    torch.cuda.manual_seed(42)

import pickle
from tqdm import tqdm
import numba

from numba import guvectorize




pairwise_2D = np.load('ll_pairwise_499500.npy') # load

pairwise_2D = pairwise_2D.astype('float32')

triple_3D = np.load('ll_triple_166167000.npy') # save

triple_3D = triple_3D.astype('float32')


all_zeros_ll_sample = torch.load('../ll_all_zeros.pt', map_location=torch.device('cpu'))

zeros_ll_sample = all_zeros_ll_sample[0].numpy()

ones_ll_sample = torch.load('../ll_ones_1000.pt', map_location=torch.device('cpu'))

ones_ll_sample= ones_ll_sample.numpy()

ones_ll_sample = ones_ll_sample[:,0]


all_combo_array = np.load('combo_triple_166167000.npy') # save

all_combo_array = all_combo_array.astype('int16')


result_list = np.zeros([all_combo_array.shape[0],])


@guvectorize(['void(float32[:,], float32[:,:], float32[:,:,:], float32[:,], int16[:,:], float64[:,])'],
             '(n),(n,n),(n,n,n), (q),(a,p)->(a)', target='cpu')
def get_answer(single_arr, pair_arr, triple_arr, zero_arr, all_combo_array, result_list):

    for row in all_combo_array:
        temp_answer = triple_arr[row[0],row[1],row[2]] + \
        single_arr[row[0]] + \
        single_arr[row[1]] +\
        single_arr[row[2]] -\
        pair_arr[row[0],row[1]] -\
        pair_arr[row[0],row[2]] - \
        pair_arr[row[1],row[2]] - zero_arr

        result_list[row] = temp_answer


 
start = time.time()

get_answer(ones_ll_sample, pairwise_2D, triple_3D, zeros_ll_sample, all_combo_array, result_list)

end = time.time()

print(end - start)











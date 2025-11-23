#!/usr/bin/env python3

import itertools
import numpy as np
import sys
import time

import numba
from numba import prange
import math

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import sparse

sys.path.append('/home/ycks3/CDT_project/python_package')
from EinsumNetwork import Graph, EinsumNetwork

#########################################################

nComb = math.comb(1000,3)
@numba.njit
def genComb_x3(arr):
    n = arr.size
    # nComb = combCount(n, 3)
    out = np.empty((nComb, 3), dtype=arr.dtype)
    a, b, c = 0, 1, 2
    arr_a = arr[a]
    arr_b = arr[b]
    for cur in range(nComb):
        out[cur, 0] = arr_a
        out[cur, 1] = arr_b
        out[cur, 2] = arr[c]
        if c < n - 1:
            c += 1
        else:
            if b < n - 2:
                b, c = b + 1, b + 2
                arr_b = arr[b]
            else:
                a, b, c = a + 1, a + 2, a + 3
                arr_a = arr[a]
                arr_b = arr[b]
    return out

@numba.njit
def make_combo_quad(combo, chunk_num, split=40):

    x = np.arange(combo.shape[0])

    chunk_indices = np.array_split(x, split)

    size = chunk_indices[chunk_num].shape[0]

    array_zeros = np.zeros((size, 1000),dtype=np.int8)

    combo_chunk = combo[chunk_indices[chunk_num],:]

    for index in prange(size):
        
        array_zeros[index,combo_chunk[index,:]] = 1

    chunk_num = chunk_num + 1

    if chunk_num == split:
        return False, chunk_num, array_zeros, combo_chunk
    else:
        return True, chunk_num, array_zeros, combo_chunk



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
    # einet.to(device)

    dir_best_model = '/home/ycks3/CDT_project/real_training/EM_14Feb/best_models/best_model_20Feb.pt'
    res_dict = torch.load(dir_best_model)

    einet.load_state_dict(res_dict['model_state_dict'])
    einet.eval()
    einet.compile()

    return einet

class myDataset(Dataset):

    def __init__(self, my_data):
        self.len = my_data.shape[0]
        self.data = torch.from_numpy(my_data).to(device)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len


def model_eval(array_zeros, combo, batch_size=10000):
    
    tensor = torch.from_numpy(array_zeros).to(device)
    
    ll_sample = EinsumNetwork.eval_loglikelihood_batched_autocast(einet, tensor,batch_size=batch_size)   ### batch 

    data = ll_sample.detach().cpu().numpy()[:,0].astype(np.float16) ### converting to float16
    
    matrix = sparse.COO(combo.T, data, shape=(1000,1000,1000))

    return matrix


##########################################################

print('number of gpu')
print(torch.cuda.device_count())

##########################################################

print('making combo')
start = time.time()

res_combo = genComb_x3(np.arange(1000))

end = time.time()

print(end - start)
print(f'number of combo: {res_combo.shape}')

###########################################################

print('make parallel model')

einet = load_model_eval()
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# einet = nn.DataParallel(einet)

device = 'cuda'
einet.to(device)

###########################################################

print('looping right now')

repeat = True

chunk_num = 0

combined_matrix='string'

saved_dir='saved_sparse_mat'

while repeat:

    start = time.time()
    repeat, chunk_num, matrix, combo_chunk = make_combo_quad(res_combo, chunk_num)

    res_mat = model_eval(matrix, combo_chunk, batch_size=10000)

    print('matrix looks like this')
    print(res_mat.shape)
    
    print('saving right now')
    sparse.save_npz(f"{saved_dir}/triple_joint_sparse_{chunk_num-1}.npz", res_mat)

    # if type(combined_matrix) == str:
    #     combined_matrix = res_mat
    # else:
    #     combined_matrix = sparse.elemwise(np.add, combined_matrix, res_mat)
    
    end = time.time()
    print(chunk_num)
    print(end-start)
    print('--------')



print('python all done')




















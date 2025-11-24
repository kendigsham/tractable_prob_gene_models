import itertools
import numpy as np
import numba
import sys
import time
import torch
from numba import prange

sys.path.append('/path/to/CDT_project/python_package')

from EinsumNetwork import Graph, EinsumNetwork

import math

nComb = math.comb(1000,3)

nComb

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

start = time.time()

res_combo = genComb_x3(np.arange(1000))

end = time.time()

print(end - start)

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

print('looping right now')

repeat = True

chunk_num = 0

while repeat:

    start = time.time()
    repeat, chunk_num, matrix, combo_chunk = make_combo_quad(res_combo, chunk_num)
    end = time.time()
    print(chunk_num)
    print(end-start)
    print('--------')

    
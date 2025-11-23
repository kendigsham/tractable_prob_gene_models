
import numpy as np
import time
import numba
from numba import prange, njit, jit
import math


@njit(inline='always')
def genComb_generic(arr, r, nComb):
    n = arr.size
    out = np.empty((nComb, r), dtype=arr.dtype)
    idx = np.empty(r, dtype=np.int32)

    for i in range(r):
        idx[i] = i

    i = r - 1

    cur = 0
    while idx[0] < n - r + 1:
        while i > 0 and idx[i] == n - r + i:
            i -= 1
        for j in range(r):
            out[cur, j] = arr[idx[j]]
        cur += 1
        idx[i] += 1
        while i < r - 1:
            idx[i + 1] = idx[i] + 1
            i += 1
    return out


numbers = np.arange(1000)

start = time.time()

array = genComb_generic(numbers,4,math.comb(1000,4))

end = time.time()

print(end-start)

print('shape of array')
print(array.shape)

print('python all done')

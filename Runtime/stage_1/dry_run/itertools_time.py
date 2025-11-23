import itertools
import numpy as np
import time

numbers = list(range(1000))
LOG_EVERY_N = 1000000
iterator = itertools.combinations(numbers,3)

def make_combo_quad(chunk_num,iterator, maximum_dim = 5000000):
    
    results=np.zeros([maximum_dim,3],dtype=np.int16)

    array_zeros = np.zeros((maximum_dim, 1000),dtype=np.int8)


    index =0
    for index, temp_combo in enumerate(iterator):
        # process the object from the iterator

        # if (index % LOG_EVERY_N) == 0:
        #     print(f'iteration = {index}')

        results[index,0] = temp_combo[0]
        results[index,1] = temp_combo[1]
        results[index,2] = temp_combo[2]

        array_zeros[index,temp_combo] = 1

        index+=1

        
        if index == maximum_dim:
            # matrix = model_eval(array_zeros, results)
            break

    if index < maximum_dim:

        array_zeros = array_zeros[:index,:]

        results = results[:index,:]

        # matrix = model_eval(array_zeros, results)

        return False, chunk_num+1#, matrix
    else:
        print('*********************')
        print(f'chunk_num = {chunk_num}')
        return True, chunk_num+1 #, matrix
        
print('looping right now')
repeat = True

chunk_num = 0


while repeat:

    start = time.time()
    repeat, chunk_num  = make_combo_quad(chunk_num = chunk_num, iterator=iterator)
    end = time.time()

    print(chunk_num)
    print(end-start)
    print('-----------')


    
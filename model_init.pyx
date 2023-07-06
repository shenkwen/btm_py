cimport cython
import numpy as np

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def model_init(unsigned short[:, :] biterm_arr, 
               int n_topics, int n_tokens, int n_biterm): 
    # initiate two key ndarrays of the model: nb_z and nwz
    cdef unsigned int[:] nb_z = np.zeros(n_topics, dtype=np.uint32)
    cdef unsigned int[:, :] nwz = np.zeros((n_topics, n_tokens), dtype=np.uint32)
    cdef Py_ssize_t i
    cdef unsigned short w1, w2, rand_k

    for i in range(n_biterm):
        w1 = biterm_arr[i, 0]
        w2 = biterm_arr[i, 1]
        rand_k = biterm_arr[i, 2]

        nb_z[rand_k] += 1 
        nwz[rand_k, w1] += 1
        nwz[rand_k, w2] += 1

    return biterm_arr, nb_z, nwz

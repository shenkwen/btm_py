cimport cython
import numpy as np


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def comp_pzd(unsigned short[:, :] biterm_arr, float[:, :] pw_z, float[:] pz,
               int n_topics, int n_biterm):

    cdef Py_ssize_t n, i
    cdef float[:] pz_b = np.zeros(n_topics, dtype=np.float32)
    cdef float[:] pz_d = np.zeros(n_topics, dtype=np.float32)
    cdef float mid_val
    cdef float pzb_sum  # used for normalize
    cdef unsigned short w1, w2

    for n in range(n_biterm):
        w1 = biterm_arr[n, 0]
        w2 = biterm_arr[n, 1]

        # comp pz_b & pzb_sum
        pzb_sum = 0.0
        for i in range(n_topics):
            mid_val = pz[i] * pw_z[i, w1] * pw_z[i, w2]
            pz_b[i] = mid_val  # comp pz_b
            pzb_sum += mid_val  # add to pzb_sum

        # normalize pz_b, add to pz_d & pzd_sum
        for i in range(n_topics):
            pz_d[i] += pz_b[i] / pzb_sum  # add to pz_d

    # normalize pz_d
    for i in range(n_topics):
        pz_d[i] /= n_biterm

    return pz_d

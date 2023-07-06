cimport cython

@cython.cdivision(True)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def biterm_iter(unsigned short[:, :] biterm_arr, unsigned int[:, :] nwz, unsigned int[:] nb_z,
                float[:] pz, double[:] rand_float, float[:] pw_b,
                int n_topics, int n_tokens, int n_biterm, float alpha, float beta, int has_b):

    cdef Py_ssize_t n, i
    cdef Py_ssize_t new_topic = 0
    cdef unsigned short w1, w2, old_topic

    cdef double u
    cdef float pz_max

    # mid val
    cdef float pw1k, pw2k, pk

    for n in range(n_biterm):
        w1 = biterm_arr[n, 0]
        w2 = biterm_arr[n, 1]
        old_topic = biterm_arr[n, 2]

        # reset biterm topic
        nb_z[old_topic] -= 1
        nwz[old_topic, w1] -= 1
        nwz[old_topic, w2] -= 1
        assert nb_z[old_topic] >= 0
        assert nwz[old_topic, w1] >= 0
        assert nwz[old_topic, w2] >= 0

        # compute p(z|b)
        for i in range(n_topics):
            pw1k = (nwz[i, w1] + beta) / (2 * nb_z[i] + n_tokens * beta)
            pw2k = (nwz[i, w2] + beta) / (2 * nb_z[i] + 1 + n_tokens * beta)
            pk = (nb_z[i] + alpha) / (n_biterm + n_topics * alpha)
            pz[i] = pw1k * pw2k * pk
        if has_b == 1:
            pk = (nb_z[0] + alpha) / (n_biterm + n_topics * alpha)
            pz[0] = pw_b[w1] * pw_b[w2] * pk

        # sample topic for biterm b
        for i in range(1, n_topics):
            pz[i] += pz[i - 1]  # 累积概率
        u = rand_float[n]  # (0, 1)随机数
        pz_max = u * pz[n_topics - 1]
        for new_topic in range(n_topics):
            if pz[new_topic] >= pz_max:
                break

        # re-assign topic to biterm
        biterm_arr[n, 2] = new_topic
        nb_z[new_topic] += 1
        nwz[new_topic, w1] += 1
        nwz[new_topic, w2] += 1

    return biterm_arr, nwz, nb_z

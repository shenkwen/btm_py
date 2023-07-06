# -*- coding:utf-8 -*-


"""
BTM model: using original matrix(not sparse matrix)
"""


import os
import sys
import logging
import array

import numpy as np

from .model_init import model_init
from .biterm_iter import biterm_iter
from .infer_sumb import comp_pzd


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)-8s: %(message)s')


def to_biterm_list(term_list, win=15):
    """transform term_list to biterm_list
    """
    term_num = len(term_list)  # length of terms in qtext

    # 在窗口长度(win)内, 组成biterm
    biterm_list = []
    for i in range(term_num - 1):
        for j in range(i + 1, min(i + win, term_num)):
            t1, t2 = term_list[i], term_list[j]  # biterm
            if t1 != t2:
                biterm_list.extend([t1, t2])

    return biterm_list


class BitermTopicModel(object):
    """
    Biterm Topic Model.

    Parameters
    --------------------
    n_topics : int, default: 100
        number of topics.

    n_tokens: int, equal to the length of word_list *************
        length of vocabulary list.

    alpha & eta: float, default: 1/n_topics
        hyperparameters that affect sparsity of the document-topic
        (theta) and topic-word (lambda) distributions.

    win: int, default: 10
        length of the window within which biterms are constructed.

    n_iter: int, default: 5
        iteration times

    save_step: int, default: 2

    Attributes
    --------------------
    pw_z: matrix of topic_word distributions
    pz: vector of topic probalicity

    """

    def __init__(self, n_topics=100, n_tokens=None, alpha=None, beta=0.05,
                 win=15, n_iter=5, save_step=2, has_background=False,
                 random_seed=None, token_list=None):
        self.n_topics = n_topics
        self.n_tokens = n_tokens

        self.alpha = 50 / n_topics if alpha is None else alpha
        self.beta = beta
        self.win = win

        self.n_iter = n_iter
        self.save_step = save_step
        self.has_background = int(has_background)  # ***********
        self.random_seed = random_seed
        self.token_list = token_list

    def fit(self, input_file, output_dir):
        """train model
        """
        logging.info('start training model')
        total_biterm, pw_b = self._load_docs(input_file)  # load train data
        total_biterm = self._biterms_init(total_biterm)  # initiate all biterms
        total_biterm, nb_z, nwz = self._model_init(total_biterm)  # model initiate

        # start iteration
        for i in range(self.n_iter):
            logging.info('iter %s' % (i + 1))

            # prepare for cython func
            rand_float = np.random.random(self.n_biterm)
            pz = np.zeros(self.n_topics, dtype=np.float32)

            # iteration all biterms using cython code
            total_biterm, nwz, nb_z = biterm_iter(
                    total_biterm, nwz, nb_z, pz, rand_float, pw_b,
                    self.n_topics, self.n_tokens, self.n_biterm, self.alpha, self.beta,
                    self.has_background)

            if (i + 1) % self.save_step == 0 and (i + 1) != self.n_iter:
                self._comp_pz(nb_z)
                self._comp_pwz(nwz, nb_z)
                self.save_model(output_dir, iter_num=i + 1)

        logging.info('iteration finish')

        # iteration finished: convert memoryview to ndarray
        self._comp_pz(nb_z)
        self._comp_pwz(nwz, nb_z)
        self.save_model(output_dir, iter_num=i + 1)

    def _load_docs(self, input_file):
        """load docs and return data for training model, including:
            (1) all tokens' prior distribution
            (2) biterms in form of 1-d array

            Parameters
            --------------------
            input_file: docs directory

            Return
            --------------------
            pw_b: numpy.array, all tokens' prior distribution
            total_biterm: numpy.array, all biterms derived from docs
        """

        # initialize
        # ====================
        pw_b = np.zeros(self.n_tokens, dtype=np.float32)
        # initiate an empty array.array to store all biterms
        # 1-d array of unsiged short and then will be transformed to 2-d numpy.array
        total_biterm = array.array('H')

        # read docs
        # ====================
        with open(input_file) as f:
            for line in f:
                wid_list = line.rstrip().split(' ')
                wid_list = list(map(int, wid_list))
                # update pw_b
                for wid in wid_list:
                    pw_b[wid] += 1
                # update total_biterm
                biterm_list = to_biterm_list(wid_list, self.win)
                total_biterm.fromlist(biterm_list)
        logging.info('load docs finish')

        pw_b /= pw_b.sum()  # normalize pw_b
        return total_biterm, pw_b

    def _biterms_init(self, total_biterm):
        """convert total_biterm to a 2-d numpy.array of which each row is a biterm
           add a random topic to each biterm
        """
        # convert to a 2-d numpy.array
        total_biterm = np.array(total_biterm, dtype=np.uint16).reshape(-1, 2)
        self.n_biterm = total_biterm.shape[0]  # total num of biterms

        # add random topics to each row
        np.random.seed(self.random_seed)
        rand_topic = np.random.randint(0, self.n_topics, (self.n_biterm, 1), dtype=np.uint16)  # generate random topics
        total_biterm = np.concatenate((total_biterm, rand_topic), axis=1)

        logging.info('initiate biterms finish, biterms number: %s' % self.n_biterm)
        return total_biterm

    def _model_init(self, total_biterm):
        """
        initiate two key element of model (using cython code)
        """
        total_biterm, nb_z, nwz = model_init(
                total_biterm, self.n_topics, self.n_tokens, self.n_biterm)
        return total_biterm, nb_z, nwz

    def _comp_pz(self, nb_z):
        """compute p_z using nb_z
        """
        nb_z = np.array(nb_z, dtype=np.float32)
        self.p_z = nb_z / nb_z.sum()

    def _comp_pwz(self, nwz, nb_z):
        """compute pw_z using nwz & nb_z
        """
        nwz = np.array(nwz, dtype=np.float32)
        nb_z = np.array(nb_z, dtype=np.float32).reshape(self.n_topics, 1)
        self.pw_z = (nwz + self.beta) / (2 * nb_z + self.n_tokens * self.beta)

    def save_model(self, output_dir, iter_num):
        """save model: save matrix pw_z & pz
        """
        logging.info('save model begin')
        np.savetxt('%s/k%s.pz' % (output_dir, self.n_topics), self.p_z)
        np.savetxt('%s/k%s.pw_z' % (output_dir, self.n_topics), self.pw_z)
        if self.token_list is not None:
            self.save_topics(output_dir, self.token_list, iter_num)
            # save token
            with open('%s/k%s.token' % (output_dir, self.n_topics), 'w') as f:
                for token in self.token_list:
                    f.write('%s\n' % token)
        logging.info('save model finish')

    def save_topics(self, output_dir, token_list, iter_num, n_terms=10):
        """print out each topic's top terms with highest probability.
        """
        topics_info = []
        for i in range(self.n_topics):
            pw_argsort = self.pw_z[i].argsort()[::-1]  # get the indices of one topic's sorted terms in reverse order
            top_words = ' '.join([token_list[j] for j in pw_argsort[:5]])  # top 5 terms
            # top 'n_terms' terms and their probability
            top_words_prob = ' '.join(['%s:%.2f%%' % (token_list[j], self.pw_z[i, j] * 100)
                                       for j in pw_argsort[:n_terms]])
            topics_info.append((self.p_z[i], i, top_words, top_words_prob))

        # write to file
        output_file = os.path.join(output_dir, 'k%s.topic%s' % (self.n_topics, iter_num))
        with open(output_file, 'w') as f:
            f.write('\t'.join(['p(z)', 'id', 'top_words', 'top_words_prob']))  # print header
            f.write('\n')
            for val in sorted(topics_info, reverse=True):  # print topics according p(z) value in reverse order
                f.write('\t'.join(map(str, val)))
                f.write('\n')

    def _comp_pzd_sumb(self, term_list):
        """compute pz_d vector using method 'sumb'

        Parameter:
        --------------------
        term_list: list of terms(token id)

        Return:
        --------------------
        pz_d: vector of pz_d
        """
        biterm_list = to_biterm_list(term_list) if len(term_list) > 1 else []

        # biterm_list为空(两种情况: term_list长度为1；term_list里元素相同)
        # ====================
        if not biterm_list:
            token_vec = self.pw_z[:, term_list[0]]
            pz_d = self.p_z * token_vec
            pz_d /= pz_d.sum()
            return pz_d

        # biterm_list不为空
        # ====================
        biterm_arr = np.array(biterm_list, dtype=np.uint16).reshape(-1, 2)
        n_biterm = biterm_arr.shape[0]
        pz_d = comp_pzd(biterm_arr, self.pw_z, self.p_z, self.n_topics, n_biterm)
        pz_d = np.array(pz_d)
        return pz_d

    def infer(self, term_list, method='sumb', top_num=3):
        """
        infer
        """
        # compute pz_d
        result = {}
        if method == 'sumb':
            pz_d = self._comp_pzd_sumb(term_list)

        # print top topics and probability
        pzd_argsort = pz_d.argsort()[::-1]
        result = dict([(idx, pz_d[idx]) for idx in pzd_argsort[:top_num]])

        return result

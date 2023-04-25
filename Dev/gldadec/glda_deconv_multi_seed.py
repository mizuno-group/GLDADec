#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 07:53:41 2022

Allow overlapping marker genes in multiple topics

@author: docker
"""
from __future__ import absolute_import, division, unicode_literals  # noqa
import logging

import numpy as np
from  tqdm import tqdm

from Dev.gldadec import _lda_basic
from Dev.gldadec import utils
import random

logger = logging.getLogger('guidedlda')

from pathlib import Path
BASE_DIR = Path(__file__).parent
print(BASE_DIR)
print("!! GuidedLDA Decovnolution Multiple Seed !!")

class GLDADeconvMS():
    """Guided Latent Dirichlet allocation using collapsed Gibbs sampling

    Parameters
    ----------
    n_topics : int
        Number of topics

    n_iter : int, default 2000
        Number of sampling iterations

    alpha : float, default 0.1
        Dirichlet parameter for distribution over topics

    eta : float, default 0.01
        Dirichlet parameter for distribution over words

    random_state : int or RandomState, optional
        The generator used for the initial topics.

    Attributes
    ----------
    `components_` : array, shape = [n_topics, n_features]
        Point estimate of the topic-word distributions (Phi in literature)
    `topic_word_` :
        Alias for `components_`
    `word_topic_` : array, shape = [n_features, n_topics]
        Point estimate of the word-topic distributions
    `nzw_` : array, shape = [n_topics, n_features]
        Matrix of counts recording topic-word assignments in final iteration.
    `ndz_` : array, shape = [n_samples, n_topics]
        Matrix of counts recording document-topic assignments in final iteration.
    `doc_topic_` : array, shape = [n_samples, n_features]
        Point estimate of the document-topic distributions (Theta in literature)
    `nz_` : array, shape = [n_topics]
        Array of topic assignment counts in final iteration.

    Examples
    --------
    >>> import numpy
    >>> X = numpy.array([[1,1], [2, 1], [3, 1], [4, 1], [5, 8], [6, 1]])
    >>> import lda
    >>> model = lda.LDA(n_topics=2, random_state=0, n_iter=100)
    >>> model.fit(X) #doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    LDA(alpha=...
    >>> model.components_
    array([[ 0.85714286,  0.14285714],
           [ 0.45      ,  0.55      ]])
    >>> model.loglikelihood() #doctest: +ELLIPSIS
    -40.395...

    References
    ----------
    Blei, David M., Andrew Y. Ng, and Michael I. Jordan. "Latent Dirichlet
    Allocation." Journal of Machine Learning Research 3 (2003): 993–1022.

    Griffiths, Thomas L., and Mark Steyvers. "Finding Scientific Topics."
    Proceedings of the National Academy of Sciences 101 (2004): 5228–5235.
    doi:10.1073/pnas.0307752101.

    Wallach, Hanna, David Mimno, and Andrew McCallum. "Rethinking LDA: Why
    Priors Matter." In Advances in Neural Information Processing Systems 22,
    edited by Y.  Bengio, D. Schuurmans, J. Lafferty, C. K. I. Williams, and A.
    Culotta, 1973–1981, 2009.

    Buntine, Wray. "Estimating Likelihoods for Topic Models." In Advances in
    Machine Learning, First Asian Conference on Machine Learning (2009): 51–64.
    doi:10.1007/978-3-642-05224-8_6.

    """

    def __init__(self, n_topics, n_iter=2000, alpha=0.01, eta=0.01, random_state=123, refresh=10, verbose=True):
        self.n_topics = n_topics
        self.n_iter = n_iter
        self.alpha = alpha
        self.eta = eta
        # if random_state is None, check_random_state(None) does nothing
        # other than return the current numpy RandomState
        self.random_state = random_state
        self.refresh = refresh
        self.verbose = verbose

        if alpha <= 0 or eta <= 0:
            raise ValueError("alpha and eta must be greater than zero")

        # random numbers that are reused
        """
        rng = utils.check_random_state(random_state)
        if random_state:
            random.seed(random_state)
        self._rands = rng.rand(1024**2 // 8)  # 1MiB of random variates
        """
        rng = np.random.RandomState(random_state)
        self._rands = rng.rand(1024**2 // 8)

        # configure console logging if not already configured
        if len(logger.handlers) == 1 and isinstance(logger.handlers[0], logging.NullHandler):
            logging.basicConfig(level=logging.INFO)
    
    def fit(self, X, seed_topics={}, initial_conf=1.0, seed_conf=0.9, other_conf=0.1, fix_seed_k=True, seed_k:list=[]):
        """Fit the model with X.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            Training data, where n_samples in the number of samples
            and n_features is the number of features. Sparse matrix allowed.
        
        seed_topics : dicts
            values : {126: [1], 161: [1], 216: [1, 2, 3, 6], 23: [1], 146: [1]}
            216 th gene is regarded as marker in 4 types of cells.
        """
        self.X = X
        self._fit(X, seed_topics=seed_topics, initial_conf=initial_conf, seed_conf=seed_conf, other_conf=other_conf, 
                  fix_seed_k=fix_seed_k, seed_k=seed_k)
        return self

    def _fit(self, X, seed_topics, initial_conf, seed_conf, other_conf, fix_seed_k=True, seed_k:list=[]):
        """Fit the model to the data X

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            Training vector, where n_samples in the number of samples and
            n_features is the number of features. Sparse matrix allowed.
            
        free_overlap: bool
            Only cell-specific genes are considered true seeds and fixed under the influence of seed_conf parameter.
            (e.g.)
            126: [1], 161: [1], 216: [1, 2, 3, 6], 23: [1]
                fix 126, 161, 23 and set free 216
        fix_seed_k: bool
            Ignore the seed_k info and regard the cell specific marker genes as anchor.
        """
        #random_state = utils.check_random_state(self.random_state)
        rng = np.random.RandomState(self.random_state)
        rands = self._rands.copy()

        self._initialize(X, seed_topics, initial_conf) # initialization
        # define seed_k (gene recognized by seed_conf parameter)
        if fix_seed_k:
            if len(seed_k)==0:
                seed_k = np.array([len(X.T)+100])
                if self.verbose:
                    print("seed_k :",0,"genes")
            else:
                seed_k = np.array(seed_k)
                if self.verbose:
                    print("seed_k :",len(seed_k),"genes")
        # regard cell specific marker genes as anchor genes
        else:
            seed_k = []
            for i,k in enumerate(seed_topics):
                if len(seed_topics.get(k))==1:
                    seed_k.append(k)
                else:
                    pass
            seed_k = np.array(seed_k)
            if self.verbose:
                print("seed_k :",len(seed_k),"genes")
        D, W = X.shape
        if self.verbose:
            for it in tqdm(range(self.n_iter)):
                # FIXME: using numpy.roll with a random shift might be faster
                rng.shuffle(rands)
                if it % self.refresh == 0:
                    ll = self.loglikelihood()
                    logger.info("<{}> log likelihood: {:.0f}".format(it, ll))
                    # keep track of loglikelihoods for monitoring convergence
                    self.loglikelihoods_.append(ll)
                self._sample_topics(rands,seed_k,seed_conf,other_conf)
        else:
            for it in range(self.n_iter):
                # FIXME: using numpy.roll with a random shift might be faster
                rng.shuffle(rands)
                if it % self.refresh == 0:
                    ll = self.loglikelihood()
                    logger.info("<{}> log likelihood: {:.0f}".format(it, ll))
                    # keep track of loglikelihoods for monitoring convergence
                    self.loglikelihoods_.append(ll)
                self._sample_topics(rands,seed_k,seed_conf,other_conf)
                
        ll = self.loglikelihood()
        logger.info("<{}> log likelihood: {:.0f}".format(self.n_iter - 1, ll))
        # note: numpy /= is integer division
        self.components_ = (self.nzw_ + self.eta).astype(float)
        self.components_ /= np.sum(self.components_, axis=1)[:, np.newaxis]
        self.topic_word_ = self.components_

        self.word_topic_ = (self.nzw_ + self.eta).astype(float)
        self.word_topic_ /= np.sum(self.word_topic_, axis=0)[np.newaxis, :]
        self.word_topic_ = self.word_topic_.T
        self.doc_topic_ = (self.ndz_ + self.alpha).astype(float)
        self.doc_topic_ /= np.sum(self.doc_topic_, axis=1)[:, np.newaxis]

        # delete attributes no longer needed after fitting to save memory and reduce clutter
        #del self.WS
        #del self.DS
        #del self.ZS
        return self
    
    def get_topicword(self):
        components_ = (self.nzw_ + self.eta).astype(float)
        components_ /= np.sum(components_, axis=1)[:, np.newaxis]
        topic_word_ = components_
        return topic_word_

    def _initialize(self, X, seed_topics, initial_confidence):
        """Initialize the document topic distribution.
        topic word distribution, etc.

        Parameters
        ----------
        seed_topics: type=dict, value={126: [1], 161: [1], 216: [1, 2, 3, 6], 23: [1], 146: [1]}
        """
        self.seed_topics = seed_topics
        D, W = X.shape
        N = int(X.sum())
        n_topics = self.n_topics
        n_iter = self.n_iter
        logger.info("n_documents: {}".format(D))
        logger.info("vocab_size: {}".format(W))
        logger.info("n_words: {}".format(N))
        logger.info("n_topics: {}".format(n_topics))
        logger.info("n_iter: {}".format(n_iter))

        self.beta = 0.1
        self.nzw_ = nzw_ = np.zeros((n_topics, W), dtype=np.intc) # + self.beta
        self.ndz_ = ndz_ = np.zeros((D, n_topics), dtype=np.intc) # + self.alpha
        self.nz_ = nz_ = np.zeros(n_topics, dtype=np.intc)# + W * self.beta

        self.WS, self.DS = WS, DS = utils.matrix_to_lists(X)
        self.ZS = ZS = np.empty_like(self.WS, dtype=np.intc)
        np.testing.assert_equal(N, len(WS))

        # seeded Initialization
        for i in range(N):
            w, d = WS[i], DS[i]
            if w not in seed_topics:
                continue
            # check if seeded initialization
            if w in seed_topics and random.random() < initial_confidence: # 0 <= random.random() < 1
                topic_candi = seed_topics[w]
                # FIXME: do not fix the random state
                #random.seed(self.random_state)
                z_new = random.sample(topic_candi,1)[0]
            else:
                z_new = i % n_topics
            ZS[i] = z_new
            ndz_[d, z_new] += 1
            nzw_[z_new, w] += 1
            nz_[z_new] += 1

        # Non seeded Initialization
        for i in range(N):
            w, d = WS[i], DS[i]
            if w in seed_topics:
                continue
            z_new = i % n_topics
            ZS[i] = z_new
            ndz_[d, z_new] += 1
            nzw_[z_new, w] += 1
            nz_[z_new] += 1
        
        self.loglikelihoods_ = []
        self.nzw_ = nzw_.astype(np.intc)
        self.ndz_ = ndz_.astype(np.intc)
        self.nz_ = nz_.astype(np.intc)
        
        self.initial_freq = nzw_.astype(np.intc)
        

    def purge_extra_matrices(self):
        """Clears out word topic. document topic. and internal matrix.
        Once this method is used. don't call fit_transform again.
        Just use the model for predictions.
        """
        del self.topic_word_
        del self.word_topic_
        del self.doc_topic_
        del self.nzw_
        del self.ndz_
        del self.nz_

    def loglikelihood(self):
        """
        Calculate complete log likelihood, log p(w,z)

        Formula used is log p(w,z) = log p(w|z) + log p(z)
        """
        nzw, ndz, nz = self.nzw_, self.ndz_, self.nz_
        alpha = self.alpha
        eta = self.eta
        nd = np.sum(ndz, axis=1).astype(np.intc)
        return _lda_basic._loglikelihood(nzw, ndz, nz, nd, alpha, eta)
    

    def _sample_topics(self, rands, seed_keys, seed_conf, other_conf):
        """ Samples all topic assignments. Called once per iteration.
        
        seed_key : [194 117 147  35  58 172 118]
        
        """
        n_topics, vocab_size = self.nzw_.shape
        alpha = np.repeat(self.alpha, n_topics).astype(np.float64)
        eta = np.repeat(self.eta, vocab_size).astype(np.float64)
        _lda_basic._sample_topics(self.WS, self.DS, self.ZS, self.nzw_, self.ndz_, self.nz_, alpha, eta, rands, seed_keys, seed_conf, other_conf)
    
    def perplexity(self, X=None):
        """
        Perplexity is defined as exp(-1. * log-likelihood per word)
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Document word matrix.
        Returns
        -------
        score : float
            Perplexity score.
        """
        if X == None: X = self.X
        ll = self.loglikelihood()
        N = int(X.sum())
        return np.exp(-1* ll / N)

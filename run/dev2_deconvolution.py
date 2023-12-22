#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 25 16:16:03 2022

Deconvolution core class.

@author: docker
"""
import gc
import copy
import random
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
BASE_DIR = Path(__file__).parent.parent

import sys
sys.path.append(str(BASE_DIR))

from gldadec import glda_deconv

class Deconvolution():
    def __init__(self,verbose=True):
        self.verbose = verbose
        self.marker_final_dic = None
        self.anchor_dic = None
    
    def set_marker(self,marker_final_dic:dict,anchor_dic:dict):
        """ Refine prior information with more reliable anchor genes.
        
        Args:
            marker_final_dic (dict): _description_
            anchor_dic (dict): Particularly reliable ones out of marker_final_dic.

        In the original paper, all marker genes are considered anchors, and this manipulation is invalid.
        This might be useful if there are genes among the acquired markers that you want to focus on.
        """
        self.marker_final_dic = marker_final_dic
        self.anchor_dic = anchor_dic
        
    def marker_redefine(self,ignore_empty=True):
        # reflect information of anchor_dic
        anchor_genes = list(itertools.chain.from_iterable(list(self.anchor_dic.values())))
        k_list = []
        v_list = []
        a_list = []
        for i,k in enumerate(self.marker_final_dic):
            if ignore_empty:
                if len(self.anchor_dic.get(k)) > 0:
                    tmp_v = self.marker_final_dic.get(k)
                    other = sorted(list(set(tmp_v) - set(anchor_genes)))
                    new_v = sorted(other + self.anchor_dic.get(k))
                    v_list.append(new_v)
                    k_list.append(k)
                    a_list.append(self.anchor_dic.get(k))
                else:
                    # anchors were not detected
                    pass
            else:
                tmp_v = self.marker_final_dic.get(k)
                other = sorted(list(set(tmp_v) - set(anchor_genes)))
                new_v = sorted(other + self.anchor_dic.get(k))
                v_list.append(new_v)
                k_list.append(k)
                a_list.append(self.anchor_dic.get(k))
        self.marker_dec_dic = dict(zip(k_list,v_list))
        self.anchor_dec_dic = dict(zip(k_list,a_list))
    
    def set_random(self,random_sets:list):
        """
        Random states list
        ----------
        random_sets : list
            e.g. [1448, 1632, 5913, 7927, 8614,...]
        """
        self.random_sets = random_sets
    
    def set_expression(self,df):
        """
        Set gene expression data.
        It is better to keep as many genes as possible.
        ----------
        df : DataFrame
            Genes in rows and samples in columns.
        """
        df.index = [t.upper() for t in df.index.tolist()] # re-index
        self.raw_df = df
        if self.verbose:
            a,b = self.raw_df.shape
            print(a,'genes')
            print(b,'samples')
    
    def expression_processing(self,random_n=0,random_genes=None):
        """ You can add randomly selected genes. Effects other than marker genes are considered without arbitrariness.

        Args:
            random_n (int, optional): The number of genes to be added randomly. Defaults to 0.
            random_genes (list, optional): A list of specific genes you would like to add. Defaults to None.
        """
        raw_df = copy.deepcopy(self.raw_df)
        genes = list(itertools.chain.from_iterable(list(self.marker_final_dic.values()))) # marker genes
        if random_genes is None:
            random_s = self.random_sets[0]
            random.seed(random_s)
            random_candidates = sorted(list(set(raw_df.index.tolist()) - set(genes))) # total genes - marker genes
            random_genes = random.sample(random_candidates,random_n) # choose genes from non-marker genes
        else:
            pass
        if self.verbose:
            print(len(random_genes),'genes were added at random')
        
        union = sorted(list(set(random_genes) | set(genes)))
        # NOTE: gene order might affect the estimation results
        common = sorted(list(set(raw_df.index.tolist()) & set(union)))
        final_df = raw_df.loc[common]
        self.final_int_dec = final_df.astype(int) # convert int
        self.input_mat_dec = np.array(self.final_int_dec.T,dtype='int64')

        # seed-topic preparation
        self.gene_names = [t.upper() for t in self.final_int_dec.index.tolist()]
        self.gene2id = dict((v, idx) for idx, v in enumerate(self.gene_names))
    
    def set_final_int(self,final_int):
        """
        Directly input the expression levels for the analysis.
        You can skip the 'set_expression()' >> 'expresison_processing()' with this step.
        ----------
        final_int : pd.DataFrame
                   PBMCs, 17-002  PBMCs, 17-006  ...  PBMCs, 17-060  PBMCs, 17-061
        S100A6                 7              7  ...              7              7
        AGTR1                  4              4  ...              4              4
        C10ORF99               4              4  ...              4              4
                         ...            ...  ...            ...            ...
        SLC6A14                4              4  ...              4              4
        EPHX2                  4              6  ...              4              5
        FOS                   12             11  ...             11             11

        """
        self.final_int_dec = final_int
        self.input_mat_dec = np.array(self.final_int_dec.T,dtype='int64')
        # seed-topic preparation
        self.gene_names = [t.upper() for t in self.final_int_dec.index.tolist()]
        self.gene2id = dict((v, idx) for idx, v in enumerate(self.gene_names))
    
    def seed_processing(self):
        """
        Prepare seed information for use as a guide.
        
        input_mat : np.array
            samples are in rows and genes (markers) are in columns.
            array([[7, 4, 5, ..., 4, 9, 4],
                   [7, 4, 5, ..., 5, 8, 4],
                   [6, 4, 4, ..., 4, 9, 5],
                   ...,
                   [7, 4, 4, ..., 4, 8, 4],
                   [7, 4, 5, ..., 4, 9, 4],
                   [8, 4, 4, ..., 4, 9, 4]])
        seed_topics : dict
            seed_topics: dict
                e.g.{0: [4,3],
                     1: [4],
                     2: [1],
                     3: [1,3,5],
                     4: [1],
                     5: [7]}
        seed_k : list
            [1,3,5,7,9,11,13]
        marker_final_dic : dcit
            {'B cells memory': ['AIM2', 'CR2', 'JCHAIN'],
             'B cells naive': ['BCL7A', 'CD24', 'FCER2', 'IL4R', 'PAX5', 'TCL1A'],
             'Monocytes': ['ALOX5AP','C5AR1','CCR2','CD14','CD163','CD274',...]}

        """
        if self.marker_dec_dic is None:
            raise ValueError('!! Final Marker Candidates were not defined !! --> run expression_processing()')
        # seed_topic preparation
        genes = list(itertools.chain.from_iterable(list(self.marker_dec_dic.values())))
        target = list(self.marker_dec_dic.keys())
        seed_topic_list = [self.marker_dec_dic.get(t) for t in target]
        seed_topics = {}
        finish_genes = []
        for t_id, st in enumerate(seed_topic_list):
            for gene in st:
                try:
                    if gene in finish_genes:
                        tmp = seed_topics[self.gene2id[gene]]
                        seed_topics[self.gene2id[gene]] = tmp + [t_id]
                    else:
                        seed_topics[self.gene2id[gene]] = [t_id]
                        finish_genes.append(gene)
                except:
                    # not included in target expression table
                    print(gene)
                    pass
                
        # reliable gene
        genes = list(itertools.chain.from_iterable(list(self.anchor_dec_dic.values())))
        seed_k = []
        for g in genes:
            if self.gene2id.get(g) is None:
                pass
            else:
                seed_k.append(self.gene2id.get(g))

        self.seed_topics = seed_topics
        # remove overlap
        seed_k = sorted(list(set(seed_k)))
        self.seed_k = seed_k
        
        if self.verbose:
            print("final genes:",len(self.final_int_dec))
            print('seed number:',len(self.seed_topics))
            print("seed_k:",len(self.seed_k))
    
    def conduct_deconv(self,add_topic=0,n_iter=200,alpha=0.01,eta=0.01,random_state=123,refresh=10):
        """ Single Run.

        add_topic : int
            The number of additional cells to be assumed in addition to the cells with markers. The default is 0.
        n_iter : int
            The number of iterations. The default is 200.
        alpha : float
            Parameter of LDA. The default is 0.01.
        eta : float
            Parameter of LDA. The default is 0.01.
        random_state : int
            The default is 123.
        refresh : int
            Interval for obtaining log-likelihood. The default is 10.
        """
        target = list(self.marker_dec_dic.keys())
        input_mat = self.input_mat_dec
        seed_topics = self.seed_topics
        seed_k = self.seed_k
        
        # model setting
        model = glda_deconv.GLDADeconvMS(
            n_topics=len(target)+add_topic,
            n_iter=n_iter, 
            alpha=alpha,
            eta=eta, 
            random_state=random_state,
            refresh=refresh
            )
        # perform the deconvolution model
        model.fit(input_mat,seed_topics=seed_topics,initial_conf=1.0,seed_conf=1.0,other_conf=0.0,fix_seed_k=True,seed_k=seed_k) 
        # plot log-likelihood
        ll = model.loglikelihoods_
        x = [i*refresh for i in range(len(ll))]
        plt.plot(x,ll)
        plt.xlabel('iterations')
        plt.ylabel('log-likelihood')
        plt.show()
        
        # confirm the correlation
        target_res = model.doc_topic_
        res = pd.DataFrame(target_res)
        res.index = self.final_int_dec.columns.tolist()
        if len(res.T) > len(target):
            target = target+[i+1 for i in range(add_topic)]
            res.columns = target
        else:
            res.columns = target
        
        self.deconv_res = res

        del model
        gc.collect()
    
    def ensemble_deconv(self,add_topic=0,n_iter=200,alpha=0.01,eta=0.01,refresh=10,initial_conf=1.0,seed_conf=1.0,other_conf=0.0,fix_seed_k=True,verbose=False):
        """ Ensemble learning with random numbers.

        add_topic : int
            The number of additional cells to be assumed in addition to the cells with markers. The default is 0.
        n_iter : int
            The number of iterations. The default is 200.
        alpha : float
            Parameter of LDA. The default is 0.01.
        eta : float
            Parameter of LDA. The default is 0.01.
        refresh : int
            Interval for obtaining log-likelihood. The default is 10.
        initial_conf : float
            Probability of using prior information for guiding (initialization). The default is 1.0.
        seed_conf : float
            Probability of maintaining guiding during the learning process. The default is 1.0.
        other_conf : float
            Probability of retaining unguided genes to the topic. The default is 0.0.
        fix_seed_k : bool
            Whether seed_k is used as prior information or not. If false, the cell-specific markers are automatically recognized and seed_k is generated. The default is True.

        """
        target = list(self.marker_dec_dic.keys())
        input_mat = self.input_mat_dec
        seed_topics = self.seed_topics
        seed_k = self.seed_k
        # ensemble
        total_res = []
        total_res2 = []
        gene_contribution = []
        ll_list = []
        for idx,rs in enumerate(self.random_sets):
            add_topic=add_topic
            # model setting
            model = glda_deconv.GLDADeconvMS(
                n_topics=len(target)+add_topic,
                n_iter=n_iter, 
                alpha=alpha,
                eta=eta, 
                random_state=rs,
                refresh=refresh,
                verbose=verbose
                )
            model.fit(input_mat,seed_topics=seed_topics,
                      initial_conf=initial_conf,seed_conf=seed_conf,other_conf=other_conf,fix_seed_k=fix_seed_k,seed_k=seed_k)
            # plot log-likelihood
            ll = model.loglikelihoods_
            ll_list.append(ll)
            
            # deconv res
            target_res = model.doc_topic_
            res = pd.DataFrame(target_res)
            res.index = self.final_int_dec.columns.tolist()

            gc_df = pd.DataFrame(model.word_topic_,index=self.final_int_dec.index.tolist()) # (gene, topic)
            init_df = pd.DataFrame(model.initial_freq,columns=self.final_int_dec.index.tolist()) # (topic, gene)
            
            if len(res.T) > len(target):
                new_target = target+[i+1 for i in range(add_topic)]
            else:
                new_target = target

            res.columns = new_target
            gc_df.columns = new_target
            init_df.index = new_target

            total_res.append(res)
            total_res2.append(init_df)
            gene_contribution.append(gc_df)
            if self.verbose:
                print(idx+1,end=" ")
            
            del model
            gc.collect()
        
        self.total_res = total_res
        self.total_res2 = total_res2
        self.ll_list = ll_list
        self.gene_contribution = gene_contribution

        
def main():
    BASE_DIR = '/workspace/github/GLDADec'
    SET_DIRECT = True
    raw_df = pd.read_csv(BASE_DIR+'/data/expression/mouse_dili/mouse_dili_expression.csv',index_col=0)
    final_int = pd.read_csv('/path/to/final_int.csv',index_col=0) # You can obtain the final marker via 'dev1_set_data.py'.
    marker_final_dic = pd.read_pickle('/path/to/marker_final_dic.pkl') # You can obtain the final marker via 'dev1_set_data.py'.
    random_sets = pd.read_pickle(BASE_DIR+'/data/random_info/10_random_sets.pkl')
    
    Dec = Deconvolution()
    Dec.set_marker(marker_final_dic=marker_final_dic,anchor_dic=marker_final_dic) # All marker genes are considered reliable and treated as anchors.
    Dec.marker_redefine()
    Dec.set_random(random_sets=random_sets)
    if SET_DIRECT:
        Dec.set_final_int(final_int=final_int)
    else:
        Dec.set_expression(df=raw_df)
        Dec.expression_processing(random_n=0)
    
    Dec.seed_processing()
    #Dec.conduct_deconv(add_topic=0,n_iter=200,alpha=0.01,eta=0.01,random_state=123,refresh=10)
    Dec.ensemble_deconv(add_topic=0,n_iter=200,alpha=0.01,eta=0.01,refresh=10)
    
    marker_dec_dic = Dec.marker_dec_dic
    anchor_dec_dic = Dec.anchor_dec_dic
    
    total_res = Dec.total_res
    print(total_res[0].shape)
    
if __name__ == '__main__':
    main()
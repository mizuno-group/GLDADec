#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 06:14:31 2022

@author: docker
"""
import copy
import random
import itertools
import collections
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from Dev.gldadec import utils

class SetData():
    def __init__(self,verbose=True):
        self.verbose = verbose
        self.raw_df = None
        self.marker_dic = None
        self.final_int = None
        self.input_mat = None
    
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
    
    def set_marker(self,marker_dic:dict):
        """
        Set marker list for each cell
        ----------
        marker_dic : dict
            
        """
        # convert uppercase
        new_v = []
        new_k = []
        for i,k in enumerate(marker_dic):
            if len(marker_dic.get(k)) > 0:
                tmp_v = sorted([t.upper() for t in marker_dic.get(k)])
                new_v.append(tmp_v)
                new_k.append(k)
            else:
                pass
        marker_dic2 = dict(zip(new_k,new_v))
        self.marker_dic = marker_dic2
        if self.verbose:
            print(len(self.marker_dic),'cells')
            print(len(marker_dic)-len(self.marker_dic),'cells were removed (markers were not registered)')
            
    def marker_info_processing(self,do_plot=True):
        # reflect expression data
        marker_dic = self.marker_dic
        genes = self.raw_df.index.tolist()
        new_v = []
        new_k = []
        for i,k in enumerate(marker_dic):
            marker = marker_dic.get(k)
            tmp_common = sorted(list(set(marker) & set(genes)))
            if len(tmp_common) > 0:
                tmp_v = [t.upper() for t in tmp_common]
                new_v.append(tmp_v)
                new_k.append(k)
            else:
                pass
        marker_dic3 = dict(zip(new_k,new_v))
        self.marker_dic = marker_dic3
        marker_genes = set(list(itertools.chain.from_iterable(list(self.marker_dic.values()))))
        if self.verbose:
            print('--- reflect genes in expression ---')
            print(len(self.marker_dic),'cells')
            print(len(marker_dic)-len(self.marker_dic),'cells were removed (markers were not registered)')
            print(len(marker_genes),'genes were registered')
        
        # plot the original registered marker size
        if do_plot:
            y = [len(t) for t in self.marker_dic.values()]
            x = [i for i in range(len(y))]
            plt.bar(x,y)
            plt.xticks(x,self.marker_dic.keys(),rotation=75)
            plt.title('Original Marker Size')
            plt.show()
        
        # detect cell specific markers
        count_dic = dict(collections.Counter(list(itertools.chain.from_iterable(list(self.marker_dic.values())))))
        sort_count = sorted(count_dic.items(),key=lambda x : x[1])
        unique_marker = [] # no overlap
        for t in sort_count:
            if t[1] == 1:
                unique_marker.append(t[0])
            else:
                pass
        new_v = []
        new_k = []
        for i,k in enumerate(self.marker_dic):
            tmp_v = sorted(list(set(self.marker_dic.get(k)) & set(unique_marker)))
            if len(tmp_v) > 0:
                new_v.append(tmp_v)
                new_k.append(k)
            else:
                pass
        self.spe_marker_dic = dict(zip(new_k,new_v))
        spe_marker_genes = set(list(itertools.chain.from_iterable(list(self.spe_marker_dic.values()))))
        if self.verbose:
            print('--- extract cell specific marker ---')
            print(len(self.spe_marker_dic),'cells')
            print(set(self.marker_dic.keys())-set(self.spe_marker_dic.keys()),'cells were removed (no marker after removing overlap)')
            print(len(spe_marker_genes),'genes were registered')
        
        # plot the cell specific marker size
        if do_plot:
            y = [len(t) for t in self.spe_marker_dic.values()]
            x = [i for i in range(len(y))]
            plt.bar(x,y)
            plt.xticks(x,self.spe_marker_dic.keys(),rotation=75)
            plt.title('Specific Marker Size')
            plt.show()
        
    def set_random(self,random_sets:list):
        """
        Random states list
        ----------
        random_sets : list
            e.g. [1448, 1632, 5913, 7927, 8614,...]
        """
        self.random_sets = random_sets
    
    def expression_processing(self,random_genes=None,random_n=0,specific=True,random_s=None,prior_norm=True,norm_scale=1000):
        """
        1. Determine if the markers are cell specific.
        2. Add non-marker gene at random.
        3. Process expression data into a format for analysis
        ----------
        random_n : int
            DESCRIPTION. The default is 0.
        specific : bool
            DESCRIPTION. The default is True.
        """
        if specific:
            if self.verbose:
                print('use specific markers')
            self.marker_final_dic = self.spe_marker_dic
        else:
            if self.verbose:
                print('use overlap markers')
            self.marker_final_dic = self.marker_dic
        
        genes = list(itertools.chain.from_iterable(list(self.marker_final_dic.values()))) # marker genes
        
        raw_df = copy.deepcopy(self.raw_df)
        if random_s is None:
            random_s = self.random_sets[0]
        random.seed(random_s)
        random_candidates = sorted(list(set(raw_df.index.tolist()) - set(genes))) # total genes - marker genes
        if random_genes is None:
            random_genes = random.sample(random_candidates,random_n) # choose genes from non-marker genes
            if self.verbose:
                print(len(random_genes),'genes were added at random')
        else:
            pass
        
        union = sorted(list(set(random_genes) | set(genes)))
        common = sorted(list(set(raw_df.index.tolist()) & set(union))) # fix the output gene order
        target_df = raw_df.loc[common]

        # prior information normalization
        if prior_norm:
            linear_norm = utils.freq_norm(target_df,self.marker_final_dic)
            linear_norm = linear_norm.loc[sorted(linear_norm.index.tolist())]
            final_df = linear_norm/norm_scale
        else:
            final_df = target_df/norm_scale
        self.final_int = final_df.astype(int) # convert int
        self.input_mat = np.array(self.final_int.T,dtype='int64')

        # seed-topic preparation
        gene_names = [t.upper() for t in self.final_int.index.tolist()]
        self.gene2id = dict((v, idx) for idx, v in enumerate(gene_names))
        self.random_genes = random_genes
    
    def expression_processing2(self,specific=True):
        """
        1. Determine if the markers are cell specific.
        2. Add non-marker gene at random to each topic.
        3. Process expression data into a format for analysis
        ----------
        specific : bool
            DESCRIPTION. The default is True.
        """
        if specific:
            if self.verbose:
                print('use specific markers')
            self.marker_final_dic = self.spe_marker_dic
        else:
            if self.verbose:
                print('use overlap markers')
            self.marker_final_dic = self.marker_dic
        
        marker_final_dic = copy.deepcopy(self.marker_final_dic)
        genes = list(itertools.chain.from_iterable(list(marker_final_dic.values()))) # marker genes
        raw_df = copy.deepcopy(self.raw_df)

        random_list = []
        new_list = []
        for i,k in enumerate(marker_final_dic):
            m = marker_final_dic.get(k)
            random_candidates = sorted(list(set(raw_df.index.tolist()) - set(genes))) # total genes - marker genes
            random.seed(i)
            random_gene = random.sample(random_candidates,len(m))
            m.extend(random_gene)
            new_list.append(sorted(m))
            random_list.append(random_gene)
            genes.extend(random_gene)
        new_dic = dict(zip(list(marker_final_dic.keys()), new_list))
        # FIXME: overwrite
        self.marker_final_dic = new_dic
        
        common = list(itertools.chain.from_iterable(list(new_dic.values()))) # marker genes
        final_df = raw_df.loc[common]
        self.final_int = final_df.astype(int) # convert int
        self.input_mat = np.array(self.final_int.T,dtype='int64')

        # seed-topic preparation
        gene_names = [t.upper() for t in self.final_int.index.tolist()]
        self.gene2id = dict((v, idx) for idx, v in enumerate(gene_names))
        #self.random_genes = random_genes
    
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
        if self.marker_final_dic is None:
            raise ValueError('!! Final Marker Candidates were not defined !! --> run expression_processing()')
        # seed_topic preparation
        genes = list(itertools.chain.from_iterable(list(self.marker_final_dic.values())))
        target = list(self.marker_final_dic.keys())
        seed_topic_list = [self.marker_final_dic.get(t) for t in target]
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
        genes = list(itertools.chain.from_iterable(list(self.marker_final_dic.values())))
        seed_k = []
        for g in genes:
            if self.gene2id.get(g) is None:
                #print(g)
                pass
            else:
                seed_k.append(self.gene2id.get(g))

        self.seed_topics = seed_topics
        seed_k = sorted(list(set(seed_k)))
        self.seed_k = seed_k
        
        if self.verbose:
            print("final genes:",len(self.final_int))
            print('seed number:',len(self.seed_topics))
            print("seed_k:",len(self.seed_k))

def main():
    raw_df = pd.read_csv('/mnt/AzumaDeconv/github/GLDADec/data/GSE65133/GSE65133_expression.csv',index_col=0)
    marker_dic = pd.read_pickle('/mnt/AzumaDeconv/github/GLDADec/data/domain_info/human_PBMC_CellMarker_8cell_raw_dic_v1.pkl')
    random_sets = pd.read_pickle('/mnt/AzumaDeconv/github/GLDADec/data/random_info/100_random_sets.pkl')

    SD = SetData()
    SD.set_expression(df=raw_df) 
    SD.set_marker(marker_dic=marker_dic)
    SD.marker_info_processing(do_plot=True)
    SD.set_random(random_sets=random_sets)
    SD.expression_processing(random_n=0,specific=True)
    SD.seed_processing()
    
    # Collect data to be used in later analyses
    input_mat = SD.input_mat
    final_int = SD.final_int
    seed_topics = SD.seed_topics
    marker_final_dic = SD.marker_final_dic
    
    # save
    out_path = '/mnt/AzumaDeconv/github/GLDADec/Dev/test_data/'
    pd.to_pickle(final_int,out_path+'final_int.pkl')
    pd.to_pickle(seed_topics,out_path+'seed_topics.pkl')
    pd.to_pickle(marker_final_dic,out_path+'marker_final_dic.pkl')

if __name__ == '__main__':
    main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 08:14:05 2022

Detect anchor genes for each cell from marker candidates

@author: docker
"""
import gc
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
#import pprint

import sys
from pathlib import Path
BASE_DIR = Path(__file__).parent

from gldadec import glda_deconv

class AnchorDetection():
    def __init__(self):
        self.final_int = None
        self.input_mat = None
        self.seed_topics = None
        self.seed_k = []
        self.marker_final_dic = None
    
    def set_data(self,final_int,seed_topics,seed_k,marker_final_dic,random_sets):
        """
        Set the required input information. You can collect from SetData class (dev1_set_data.py)

        Parameters
        ----------
        final_int : pd.DataFrame
                PBMCs, 17-002  PBMCs, 17-006  ...  PBMCs, 17-060  PBMCs, 17-061
        ZBTB16               4              4  ...              4              4
        CST3                11              9  ...              9             11
        CD247                9             10  ...              9             10
        ARMH1                6              6  ...              6              6
        RIC3                 4              4  ...              4              4
                       ...            ...  ...            ...            ...
        THAP7                5              4  ...              4              4
        TMEM123              9              9  ...              9              9
        SDC1                 4              4  ...              4              4
        S100A6               7              7  ...              7              7
        LMO7                 4              4  ...              4              4
        
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
        random_sets : list
            e.g. [1448, 1632, 5913, 7927, 8614,...]
        """
        self.final_int = final_int
        self.input_mat = np.array(self.final_int.T,dtype='int64')
        self.seed_topics = seed_topics
        self.marker_final_dic = marker_final_dic
        self.random_sets = random_sets
    
    def ll_monitor(self,add_topic=0,n_iter=1000,alpha=0.01,eta=0.01,random_state=123,refresh=10):
        """
        Monitoring log-likelihood transition

        Parameters
        ----------
        add_topic : int
            The number of additional cells to be assumed in addition to the cells with markers. The default is 0.
        n_iter : int
            The number of iterations. The default is 1000.
        alpha : float
            Parameter of LDA. The default is 0.01.
        eta : float
            Parameter of LDA. The default is 0.01.
        random_state : int
            The default is 123.
        refresh : TYPE, optional
            Interval for obtaining log-likelihood. The default is 10.

        """
        target = list(self.marker_final_dic.keys())
        input_mat = self.input_mat
        seed_topics = self.seed_topics
        
        # model setting
        model = glda_deconv_multi_seed.GLDADeconvMS(
            n_topics=len(target)+add_topic,
            n_iter=n_iter, 
            alpha=alpha,
            eta=eta, 
            random_state=random_state,
            refresh=refresh
            )
        
        model.fit(input_mat,seed_topics=seed_topics,initial_conf=1.0,seed_conf=0.0,other_conf=0.0,fix_seed_k=True,seed_k=[]) # free to move
        # plot log-likelihood
        self.ll = model.loglikelihoods_
        x = [i*refresh for i in range(len(self.ll))]
        plt.plot(x,model.loglikelihoods_)
        plt.xlabel('iterations')
        plt.ylabel('log-likelihood')
        plt.show()

        del model
        gc.collect()
    
    def multi_trial(self,add_topic=0,n_iter=200,alpha=0.01,eta=0.01,refresh=10):
        # anchor candidates movement
        target = list(self.marker_final_dic.keys())
        input_mat = self.input_mat
        seed_topics = self.seed_topics
        
        total_comp = []
        total_nzw = []
        for idx,rs in enumerate(self.random_sets):
            # model setting
            model = glda_deconv_multi_seed.GLDADeconvMS(
                n_topics=len(target)+add_topic,
                n_iter=n_iter,
                alpha=alpha,
                eta=eta, 
                random_state=rs,
                refresh=refresh,
                verbose=False
                )
            
            model.fit(input_mat,seed_topics=seed_topics,initial_conf=1.0,seed_conf=0.0,other_conf=0.0, fix_seed_k=True, seed_k=[]) # free to move
            
            # components (gene wide normalization)
            comp = model.components_
            comp_df = pd.DataFrame(comp,index=target,columns=self.final_int.index.tolist())
            total_comp.append(comp_df)
            
            # nzw (without any normalization)
            nzw = model.nzw_
            nzw_df = pd.DataFrame(nzw,index=target,columns=self.final_int.index.tolist())
            total_nzw.append(nzw_df)
            
            del model
            gc.collect()
            print(idx+1,end=' ')
        
        self.total_comp = total_comp
        self.total_nzw = total_nzw
    
    def abs_detection(self,total_nzw=None,do_plot=True,ratio=0.8):
        """
        Detect anchors with absolute threshold

        Parameters
        ----------
        total_nzw : list
            A list of gene contributions stores as a DataFrame.
        do_plot : bool
            The default is True.
        ratio : float
            The default is 0.8.
            
        """
        if total_nzw is None:
            total_nzw = self.total_nzw
        
        target = list(self.marker_final_dic.keys())
        base = len(total_nzw)
        anchor_res = []
        cell_summary = []
        for cell in target:
            for i,tmp_df in enumerate(total_nzw):
                genes = tmp_df.columns.tolist()
                cells = tmp_df.index.tolist()
                marker = self.marker_final_dic.get(cell)
                marker = sorted(list(set(marker) & set(genes)))
                marker_df = tmp_df[marker]
                # min-max scaling
                marker_mm = preprocessing.minmax_scale(marker_df,axis=0)
                marker_mm = pd.DataFrame(marker_mm,index=cells,columns=marker)
                if i == 0:
                    sum_summary = marker_mm
                else:
                    sum_summary = sum_summary + marker_mm
            # plot heatmap
            if do_plot:
                sns.heatmap(sum_summary)
                plt.title(cell)
                plt.show()
            # detect marker
            cell_summary.append(sum_summary)
            sum_summary = sum_summary.T
            r = sorted(sum_summary[sum_summary[cell] > base*ratio].index.tolist())
            anchor_res.append(r)
        self.anchor_dic = dict(zip(target,anchor_res))
        
        # plot the cell specific marker size
        if do_plot:
            fig,ax = plt.subplots(figsize=(6,6),dpi=100)
            y1 = [len(t) for t in self.marker_final_dic.values()]
            y2 = [len(t) for t in self.anchor_dic.values()]
            x = [i for i in range(len(y1))]
            x1 = [i-0.2 for i in range(len(y1))]
            x2 = [i+0.2 for i in range(len(y2))]
            plt.bar(x1,y1,width=0.4,label='original')
            plt.bar(x2,y2,width=0.4,label='anchor')
            plt.xticks(x,self.anchor_dic.keys(),rotation=75)
            plt.legend(loc='best')
            plt.title('Size of anchor genes')
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['top'].set_visible(False)
            plt.gca().yaxis.set_ticks_position('left')
            plt.gca().xaxis.set_ticks_position('bottom')
            ax.set_axisbelow(True)
            ax.grid(color="#ababab",linewidth=0.5)
            plt.show()
    
    def rel_detection(self,total_nzw=None,do_plot=True,fold_change=2,topn=-1):
        """
        Detect anchors with relative threshold

        Parameters
        ----------
        total_nzw : list
            A list of gene contributions stores as a DataFrame.
        do_plot : bool
            The default is True.
        fold_change : float
            The default is 2.
            
        """
        if total_nzw is None:
            total_nzw = self.total_nzw
        
        target = list(self.marker_final_dic.keys())
        self.cell_summary = []
        for cell in target:
            for i,tmp_df in enumerate(total_nzw):
                genes = tmp_df.columns.tolist()
                cells = tmp_df.index.tolist()
                marker = self.marker_final_dic.get(cell)
                marker = sorted(list(set(marker) & set(genes)))
                marker_df = tmp_df[marker]
                # min-max scaling
                marker_mm = preprocessing.minmax_scale(marker_df,axis=0)
                marker_mm = pd.DataFrame(marker_mm,index=cells,columns=marker)
                if i == 0:
                    sum_summary = marker_mm
                else:
                    sum_summary = sum_summary + marker_mm
            self.cell_summary.append(sum_summary)
            
            # plot heatmap
            if do_plot:
                sns.heatmap(sum_summary)
                plt.title(cell)
                plt.show()
        
        # detect marker with relative change
        anchor_res = []
        for i,k in enumerate(target):
            cell = target[i]
            tmp_df = self.cell_summary[i]
            tmp_base = tmp_df.loc[cell]
            tmp_rel = tmp_df / tmp_base # normalize with the target cell value
            first_second = pd.DataFrame(np.sort(tmp_rel.T.values).T[-2:,:],index=['second','first'],columns=tmp_df.columns.tolist()).T.sort_values('second')
            tmp_anchor = first_second[first_second['second'] < 1/fold_change].sort_values('second').index.tolist()
            if topn > 0:
                if len(tmp_anchor) > topn:
                    anchor_res.append(sorted(tmp_anchor[0:topn]))
                else:
                    anchor_res.append(sorted(tmp_anchor))
            else:
                anchor_res.append(sorted(tmp_anchor))

        self.anchor_dic = dict(zip(target,anchor_res))
        
        # plot the cell specific marker size
        fig,ax = plt.subplots(figsize=(6,5),dpi=100)
        y1 = [len(t) for t in self.marker_final_dic.values()]
        y2 = [len(t) for t in self.anchor_dic.values()]
        x = [i for i in range(len(y1))]
        x1 = [i-0.2 for i in range(len(y1))]
        x2 = [i+0.2 for i in range(len(y2))]
        plt.bar(x1,y1,width=0.4,label='original')
        plt.bar(x2,y2,width=0.4,label='anchor')
        plt.xticks(x,self.anchor_dic.keys(),rotation=75)
        plt.legend(loc='best')
        plt.title('Size of anchor genes')
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().yaxis.set_ticks_position('left')
        plt.gca().xaxis.set_ticks_position('bottom')
        ax.set_axisbelow(True)
        ax.grid(color="#ababab",linewidth=0.5)
        plt.show()
            
    def rel_detection_topn(self,total_nzw=None,do_plot=True,topn=10):
        """
        Detect forced anchors regardless of relative ratings.

        Parameters
        ----------
        total_nzw : list
            A list of gene contributions stores as a DataFrame.
        do_plot : bool
            The default is True.
            
        """
        if total_nzw is None:
            total_nzw = self.total_nzw
        
        target = list(self.marker_final_dic.keys())
        self.cell_summary = []
        for cell in target:
            for i,tmp_df in enumerate(total_nzw):
                genes = tmp_df.columns.tolist()
                cells = tmp_df.index.tolist()
                marker = self.marker_final_dic.get(cell)
                marker = sorted(list(set(marker) & set(genes)))
                marker_df = tmp_df[marker]
                # min-max scaling
                marker_mm = preprocessing.minmax_scale(marker_df,axis=0)
                marker_mm = pd.DataFrame(marker_mm,index=cells,columns=marker)
                if i == 0:
                    sum_summary = marker_mm
                else:
                    sum_summary = sum_summary + marker_mm
            self.cell_summary.append(sum_summary)
            
            # plot heatmap
            if do_plot:
                sns.heatmap(sum_summary)
                plt.title(cell)
                plt.show()
        
        # detect marker with relative change
        anchor_res = []
        for i,k in enumerate(target):
            cell = target[i]
            tmp_df = self.cell_summary[i]
            if len(tmp_df.T) < topn:
                anchor_res.append(sorted(tmp_df.columns.tolist())) # if less than topn, all candidates are considered as anchors
            else:
                tmp_base = tmp_df.loc[cell]
                tmp_rel = tmp_df / tmp_base # normalize with the target cell value
                first_second = pd.DataFrame(np.sort(tmp_rel.T.values).T[-2:,:],index=['second','first'],columns=tmp_df.columns.tolist()).T.sort_values('second')
                tmp_anchor = first_second.index.tolist()[0:topn]
                anchor_res.append(sorted(tmp_anchor))
                
        self.anchor_dic = dict(zip(target,anchor_res))
        
        # plot the cell specific marker size
        y1 = [len(t) for t in self.marker_final_dic.values()]
        y2 = [len(t) for t in self.anchor_dic.values()]
        x = [i for i in range(len(y1))]
        x1 = [i-0.2 for i in range(len(y1))]
        x2 = [i+0.2 for i in range(len(y2))]
        plt.bar(x1,y1,width=0.4,label='original')
        plt.bar(x2,y2,width=0.4,label='anchor')
        plt.xticks(x,self.anchor_dic.keys(),rotation=75)
        plt.legend(loc='best')
        plt.title('Anchor Size')
        plt.show()
        
    def rel_detection_legacy(self,total_nzw=None,do_plot=True,fold_change=2):
        """
        Detect anchors with relative threshold

        Parameters
        ----------
        total_nzw : list
            A list of gene contributions stores as a DataFrame.
        do_plot : bool
            The default is True.
        fold_change : float
            The default is 2.
            
        """
        if total_nzw is None:
            total_nzw = self.total_nzw
        
        target = list(self.marker_final_dic.keys())
        self.cell_summary = []
        for cell in target:
            for i,tmp_df in enumerate(total_nzw):
                genes = tmp_df.columns.tolist()
                cells = tmp_df.index.tolist()
                marker = self.marker_final_dic.get(cell)
                marker = sorted(list(set(marker) & set(genes)))
                marker_df = tmp_df[marker]
                # min-max scaling
                marker_mm = preprocessing.minmax_scale(marker_df,axis=0)
                marker_mm = pd.DataFrame(marker_mm,index=cells,columns=marker)
                if i == 0:
                    sum_summary = marker_mm
                else:
                    sum_summary = sum_summary + marker_mm
            self.cell_summary.append(sum_summary)
            
            # plot heatmap
            if do_plot:
                sns.heatmap(sum_summary)
                plt.title(cell)
                plt.show()
        
        # detect marker with relative change
        anchor_res = []
        for i,k in enumerate(target):
            cell = target[i]
            tmp_df = self.cell_summary[i]
            tmp_base = tmp_df.loc[cell]
            tmp_rel = tmp_df / tmp_base # normalize with the target cell value
            first_second = pd.DataFrame(np.sort(tmp_rel.T.values).T[-2:,:],index=['second','first'],columns=tmp_df.columns.tolist()).T
            tmp_anchor = first_second[first_second['second'] < 1/fold_change].index.tolist()
            anchor_res.append(tmp_anchor)

        self.anchor_dic = dict(zip(target,anchor_res))
        
        # plot the cell specific marker size
        if do_plot:
            y1 = [len(t) for t in self.marker_final_dic.values()]
            y2 = [len(t) for t in self.anchor_dic.values()]
            x = [i for i in range(len(y1))]
            x1 = [i-0.2 for i in range(len(y1))]
            x2 = [i+0.2 for i in range(len(y2))]
            plt.bar(x1,y1,width=0.4,label='original')
            plt.bar(x2,y2,width=0.4,label='anchor')
            plt.xticks(x,self.anchor_dic.keys(),rotation=75)
            plt.legend(loc='best')
            plt.title('Anchor Size')
            plt.show()

    
def main():
    in_path = '/mnt/AzumaDeconv/github/GLDADec/Dev/test_data/'
    final_int = pd.read_pickle(in_path+'final_int.pkl')
    seed_topics = pd.read_pickle(in_path+'seed_topics.pkl')
    marker_final_dic = pd.read_pickle(in_path+'marker_final_dic.pkl')
    random_sets = pd.read_pickle('/mnt/AzumaDeconv/github/GLDADec/data/random_info/100_random_sets.pkl')

    AD = AnchorDetection()
    AD.set_data(final_int=final_int, seed_topics=seed_topics, seed_k=[], marker_final_dic=marker_final_dic, random_sets=random_sets)
    AD.ll_monitor(add_topic=0,n_iter=1000,alpha=0.01,eta=0.01,random_state=123,refresh=10)
    AD.multi_trial(add_topic=0,n_iter=200,alpha=0.01,eta=0.01,refresh=10)

    total_comp = AD.total_comp
    total_nzw = AD.total_nzw
    out_path = '/mnt/AzumaDeconv/github/GLDADec/Dev/test_data/'
    pd.to_pickle(total_comp,out_path+'total_comp.pkl')
    pd.to_pickle(total_nzw,out_path+'total_nzw.pkl')

    total_nzw = pd.read_pickle(in_path+'total_nzw.pkl')
    total_comp = pd.read_pickle(in_path+'total_comp.pkl')

    # you can select from three methods
    #AD.abs_detection(total_nzw=total_comp,do_plot=True,ratio=0.9)
    #AD.rel_detection_topn(total_nzw=total_comp,do_plot=True,topn=10)
    AD.rel_detection(total_nzw=total_comp,do_plot=True,fold_change=2,topn=-1)

    cell_summary = AD.cell_summary
    anchor_dic = AD.anchor_dic
    pd.to_pickle(anchor_dic,out_path+'anchor_dic.pkl')

if __name__ == '__main__':
    main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 27 13:13:14 2022

@author: docker
"""
import pandas as pd
from matplotlib import colors as mcolors
tab_colors = mcolors.TABLEAU_COLORS

import sys
from pathlib import Path
BASE_DIR = Path(__file__).parent

from _utils import plot_utils, processing

class Evaluation():
    def __init__(self):
        self.total_res = None
        self.ref_df = None
    
    def set_res(self,total_res,z_norm=True):
        if z_norm:
            self.total_res = []
            for res in total_res:
                z_res = processing.standardz_sample(res) # sample wide normalization
                # print(z_res.sum())
                self.total_res.append(z_res)
        else:
            self.total_res = total_res
        self.samples = self.total_res[0].index.tolist()
        #print("samples of res",self.total_res[0].index.tolist())

        # ensemble
        sum_res = sum(self.total_res) / len(total_res)
        if z_norm:
            self.ensemble_res = processing.standardz_sample(sum_res) # sample-wide
        else:
            self.ensemble_res = sum_res
        print('cells in res :',self.ensemble_res.columns.tolist()) # TODO : add logging
        
    def set_ref(self,ref_df,z_norm=True):
        """
        ----------
        ref_df : pd.DataFrame
        cells in rows and samples in columns
        
                       Naive B  Memory B  CD8 T  ...  Gamma delta T     NK  Monocytes
        PBMCs, 17-002    12.36      2.96  22.36  ...           5.44  16.58      17.10
        PBMCs, 17-006     3.33      1.83  32.05  ...           2.28   8.90       5.70
        PBMCs, 17-019    15.09      4.07  17.86  ...          12.49  29.64       7.99
        PBMCs, 17-023    16.77      2.95   8.69  ...           0.64  17.11      17.08
        PBMCs, 17-026     8.40      4.69  27.98  ...           5.59   7.88      13.38
        PBMCs, 17-027    11.62      4.16  20.37  ...           7.55  14.18      10.47
        
        """
        if z_norm:
            z_ref = processing.standardz_sample(ref_df) # sample wide normalization
            z_ref = z_ref.loc[self.samples]
            self.ref_df = z_ref
        else:
            ref_df = ref_df.loc[self.samples]
            self.ref_df = ref_df
        #print("samples of ref",self.ref_df.index.tolist())
        print('cells in ref :',self.ref_df.columns.tolist()) # TODO : add logging
    
    #%% main
    def multi_eval(self,
                   res_names=[['B cells naive'],['T cells CD4 naive'],['T cells CD8'],['NK cells'],['Monocytes']],
                   ref_names=[['Naive B'],['Naive CD4 T'],['CD8 T'],['NK'],['Monocytes']],
                   title_list=['Naive B','Naive CD4 T','CD8 T','NK','Monocytes'],
                   target_samples = None,
                   figsize=(6,6),dpi=100,plot_size=100):
        color_list = list(tab_colors.keys())
        performance_list = []
        dec_eval_x = []
        ref_eval_y = []
        for i in range(len(res_names)):
            res_name = res_names[i]
            ref_name = ref_names[i]
            color = color_list[i]
            title = title_list[i]
            dat = plot_utils.DeconvPlot(deconv_df=self.ensemble_res,val_df=self.ref_df,dec_name=res_name,val_name=ref_name,figsize=figsize,dpi=dpi,plot_size=plot_size)
            a = dat.plot_simple_corr(color=color,title=title,target_samples=target_samples)
            dec_eval_x.append(a[1])
            ref_eval_y.append(a[2])
            performance = a[0]
            performance_list.append(list(performance.items()))
        
        self.evalxy = [dec_eval_x,ref_eval_y]
        self.performance_dic = dict(zip(title_list,performance_list))

        #dat.overlap_singles(evalxy=self.evalxy,title_list=title_list)
        dat.overlap_groups(evalxy=self.evalxy,res_names=res_names,ref_names=ref_names,title_list=['Naive B','Naive CD4 T','CD8 T','NK','Monocytes'],color_list=color_list,target_samples=target_samples)

    
    def multi_eval_multi_group(self,
                               res_names=[['B cells naive'],['T cells CD4 naive'],['T cells CD8'],['NK cells'],['Monocytes']],
                               ref_names=[['Naive B'],['Naive CD4 T'],['CD8 T'],['NK'],
                               ['Monocytes']],
                               title_list=['Naive B','Naive CD4 T','CD8 T','NK','Monocytes'],figsize=(6,6),plot_size=100,dpi=100):
        
        performance_list = []
        for i in range(len(res_names)):
            res_name = res_names[i]
            ref_name = ref_names[i]
            title = title_list[i]
            dat = plot_utils.DeconvPlot(deconv_df=self.ensemble_res,val_df=self.ref_df,dec_name=res_name,val_name=ref_name,figsize=figsize,dpi=dpi,plot_size=plot_size)
            a = dat.plot_group_corr(sort_index=[],sep=True,title=title)
            performance = a[0]
            performance_list.append(list(performance.items()))
        
        self.performance_dic = dict(zip(title_list,performance_list))
    
def main():
    target_facs = pd.read_csv('/mnt/AzumaDeconv/github/GLDADec/data/GSE65133/facs_results.csv',index_col=0)
    in_path = '/mnt/AzumaDeconv/Topic_Deconv/GuidedLDA/221027_GSE65133/221101_CellMarker/221229_threshold_impact/results/'
    total_res = pd.read_pickle(in_path + '/total_res_original.pkl')
    
    Eval = Evaluation()
    Eval.set_res(total_res=total_res,z_norm=True)
    Eval.set_ref(ref_df=target_facs,z_norm=True)
    
    #Eval.evaluate(res_name='B cells naive',ref_name='Naive B')
    Eval.multi_eval(res_names=[['B cells naive'],['B cells memory'],['T cells CD4 naive'],['T cells CD4 memory'],['T cells CD8'],['NK cells'],['Monocytes'],['T cells gamma delta']],
    ref_names=[['Naive B'],['Memory B'],['Naive CD4 T'],['Resting memory CD4 T', 'Activated memory CD4 T'],['CD8 T'],['NK'],['Monocytes'],['Gamma delta T']])
    
if __name__ == '__main__':
    main()
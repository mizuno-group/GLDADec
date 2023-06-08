# -*- coding: utf-8 -*-
"""
Created on 2023-06-07 (Wed) 10:07:04

@author: I.Azuma
"""
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

from logging import getLogger
logger = getLogger('plot_utils')

class DeconvPlot():
    def __init__(self,deconv_df,val_df,dec_name=['B cells naive'],val_name=['Naive B'],
                do_plot=True,figsize=(6,6),dpi=300,plot_size=100):
        self.deconv_df = deconv_df
        self.val_df = val_df
        self.dec_name = dec_name
        self.val_name = val_name
        self.do_plot = do_plot
        self.figsize = figsize
        self.dpi = dpi
        self.plot_size = plot_size

        self.xlabel = 'Estimated Proportion'
        self.ylabel = 'True Proportion'
        self.label_size = 20
        self.tick_size = 15

    def plot_simple_corr(self,color='tab:blue',title='Naive B'):
        """
        Correlation Scatter Plotting
        Format of both input dataframe is as follows
        Note that the targe data contains single treatment group (e.g. APAP treatment only)
        
                    B       CD4       CD8      Monocytes        NK  Neutrophils
        Donor_1 -0.327957 -0.808524 -0.768420   0.311360  0.028878     0.133660
        Donor_2  0.038451 -0.880116 -0.278970  -1.039572  0.865344    -0.437588
        Donor_3 -0.650633  0.574758 -0.498567  -0.796406 -0.100941     0.035709
        Donor_4 -0.479019 -0.005198 -0.675028  -0.787741  0.343481    -0.062349
        
        """
        total_x = self.deconv_df[self.dec_name].sum(axis=1).tolist()
        total_y = self.val_df[self.val_name].sum(axis=1).tolist()
        total_cor, pvalue = stats.pearsonr(total_x,total_y) # correlation and pvalue
        total_cor = round(total_cor,4)
        if pvalue < 0.01:
            pvalue = '{:.2e}'.format(pvalue)
        else:
            pvalue = round(pvalue,3)
        rmse = round(np.sqrt(mean_squared_error(total_x, total_y)),4)
        performance = {'R':total_cor,'P':pvalue,'RMSE':rmse}
        
        res1 = self.deconv_df[self.dec_name].sum(axis=1).tolist()
        res2 = self.val_df[self.val_name].sum(axis=1).tolist()
        
        x_min = min(min(res1),min(res2))
        x_max = max(max(res1),max(res2))
        
        if self.do_plot:
            fig,ax = plt.subplots(figsize=self.figsize,dpi=self.dpi)
            plt.scatter(res1,res2,alpha=1.0,s=self.plot_size,c=color)
            #plt.plot([0,x_max],[0,x_max],linewidth=2,color='black',linestyle='dashed',zorder=-1)
            plt.plot([x_min,x_max],[x_min,x_max],linewidth=2,color='black',linestyle='dashed',zorder=-1)
            plt.text(1.0,0.15,'R = {}'.format(str(round(total_cor,3))), transform=ax.transAxes, fontsize=15)
            plt.text(1.0,0.10,'P = {}'.format(str(pvalue)), transform=ax.transAxes, fontsize=15)
            plt.text(1.0,0.05,'RMSE = {}'.format(str(round(rmse,3))), transform=ax.transAxes, fontsize=15)
            
            #plt.legend(shadow=True)
            plt.xlabel(self.xlabel,fontsize=self.label_size)
            plt.ylabel(self.ylabel,fontsize=self.label_size)
            xlocs, _ = plt.xticks()
            ylocs, _ = plt.yticks()
            tick_min = max(0.0,min(xlocs[0],ylocs[0]))
            tick_max = max(xlocs[-1],ylocs[-1])
            step = (tick_max-tick_min)/5
            plt.xticks(np.arange(tick_min,tick_max,step),fontsize=self.tick_size)
            plt.yticks(np.arange(tick_min,tick_max,step),fontsize=self.tick_size)
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['top'].set_visible(False)
            plt.gca().yaxis.set_ticks_position('left')
            plt.gca().xaxis.set_ticks_position('bottom')
            ax.set_axisbelow(True)
            ax.grid(color="#ababab",linewidth=0.5)
            plt.title(title,fontsize=self.label_size)
            plt.show()
        else:
            pass
        return performance,res1,res2

    def plot_group_corr(deconv_df,val_df,dec_name=["CD4","CD8"],val_name=["abT"],sort_index=[],
                        do_plot=True,sep=True,title=None,do_print=False,dpi=300,plot_size=50):
        """
        Correlation Scatter Plotting
        Format of both input dataframe is as follows
        
                    B       CD4       CD8  Monocytes        NK  Neutrophils
        AN_1 -0.327957 -0.808524 -0.768420   0.311360  0.028878     0.133660
        AN_2  0.038451 -0.880116 -0.278970  -1.039572  0.865344    -0.437588
        AN_3 -0.650633  0.574758 -0.498567  -0.796406 -0.100941     0.035709
        AN_4 -0.479019 -0.005198 -0.675028  -0.787741  0.343481    -0.062349
        AP_1 -1.107050  0.574758  0.858366  -1.503722 -1.053643     1.010999
        
        """
        if title is None:
            title = str(dec_name)+" vs "+str(val_name)
        
        if len(sort_index)>0:
            drugs = sort_index
        elif sep:
            drugs = sorted(list(set([t.split("_")[0] for t in deconv_df.index.tolist()])))
        else:
            drugs = sorted(deconv_df.index.tolist())
        
        # align the index
        val_df = val_df.loc[deconv_df.index.tolist()]
        
        total_x = deconv_df[dec_name].sum(axis=1).tolist()
        total_y = val_df[val_name].sum(axis=1).tolist()
        total_cor, pvalue = stats.pearsonr(total_x,total_y) # correlation and pvalue
        total_cor = round(total_cor,4)
        if pvalue < 0.01:
            pvalue = '{:.2e}'.format(pvalue)
        else:
            pvalue = round(pvalue,3)
        rmse = round(np.sqrt(mean_squared_error(total_x, total_y)),4)
        
        if do_print:
            print(str(dec_name)+" vs "+str(val_name))
            print(total_cor,pvalue,rmse)
        
        if do_plot:
            fig,ax = plt.subplots(figsize=(6,6),dpi=dpi)
            x_min = 100
            x_max = -100
            for i,d in enumerate(drugs):
                tmp1 = deconv_df.filter(regex="^"+d+"_",axis=0)
                tmp2 = val_df.filter(regex="^"+d+"_",axis=0)
                
                res1 = tmp1[dec_name].sum(axis=1).tolist()
                res2 = tmp2[val_name].sum(axis=1).tolist()
                tmp_cor = round(np.corrcoef(res1,res2)[0][1],3)
            
                plt.scatter(res1,res2,label=d+" : "+str(tmp_cor),alpha=1.0,s=plot_size)
                
                xmin = min(min(res1),min(res2))
                if xmin < x_min:
                    x_min = xmin
                xmax = max(max(res1),max(res2))
                if xmax > x_max:
                    x_max = xmax
            
            plt.plot([x_min,x_max],[x_min,x_max],linewidth=2,color='black',linestyle='dashed',zorder=-1)
            plt.text(1.0,0.15,'R = {}'.format(str(round(total_cor,3))), transform=ax.transAxes, fontsize=15)
            plt.text(1.0,0.10,'P = {}'.format(str(pvalue)), transform=ax.transAxes, fontsize=15)
            plt.text(1.0,0.05,'RMSE = {}'.format(str(round(rmse,3))), transform=ax.transAxes, fontsize=15)
            
            #plt.legend(loc='upper center',shadow=True,fontsize=13,ncol=2,bbox_to_anchor=(.45, 1.12))
            plt.legend(loc="upper left", bbox_to_anchor=(0.95, 1),shadow=True,fontsize=13)
            plt.xlabel("Deconvolution estimated value",fontsize=15)
            plt.ylabel("Flow cytometry value",fontsize=15)
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['top'].set_visible(False)
            plt.gca().yaxis.set_ticks_position('left')
            plt.gca().xaxis.set_ticks_position('bottom')
            ax.set_axisbelow(True)
            ax.grid(color="#ababab",linewidth=0.5)
            plt.title(title,fontsize=15)
            plt.show()
        else:
            pass
        return total_x,total_y,total_cor

    def estimation_var(total_res,cell='Neutrophil',dpi=100):
        summary_df = pd.DataFrame()
        for idx,tmp_df in enumerate(total_res):
            cell_df = tmp_df[[cell]]
            cell_df.columns = [idx] # rename column
            summary_df = pd.concat([summary_df,cell_df],axis=1)

        sample_names = summary_df.index.tolist()
        data = []
        for sample in sample_names:
            data.append(list(summary_df.loc[sample]))
        
        # plot bar
        plot_multi(data=data,names=sample_names,value='Deconvolution value (%)', title=str(cell)+" estimation variance",grey=False,dpi=dpi)
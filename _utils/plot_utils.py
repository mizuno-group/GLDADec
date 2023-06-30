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

    def plot_group_corr(self,sort_index=[],sep=True,title=None):
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
            title = str(self.dec_name)+" vs "+str(self.val_name)
        
        if len(sort_index)>0:
            drugs = sort_index
        elif sep:
            drugs = sorted(list(set([t.split("_")[0] for t in self.deconv_df.index.tolist()])))
        else:
            drugs = sorted(self.deconv_df.index.tolist())
        
        # align the index
        val_df = self.val_df.loc[self.deconv_df.index.tolist()]
        
        total_x = self.deconv_df[self.dec_name].sum(axis=1).tolist()
        total_y = val_df[self.val_name].sum(axis=1).tolist()
        total_cor, pvalue = stats.pearsonr(total_x,total_y) # correlation and pvalue
        total_cor = round(total_cor,4)
        if pvalue < 0.01:
            pvalue = '{:.2e}'.format(pvalue)
        else:
            pvalue = round(pvalue,3)
        rmse = round(np.sqrt(mean_squared_error(total_x, total_y)),4)
        performance = {'R':total_cor,'P':pvalue,'RMSE':rmse}

        fig,ax = plt.subplots(figsize=(6,6),dpi=self.dpi)
        x_min = 100
        x_max = -100
        for i,d in enumerate(drugs):
            tmp1 = self.deconv_df.filter(regex="^"+d+"_",axis=0)
            tmp2 = val_df.filter(regex="^"+d+"_",axis=0)
            
            res1 = tmp1[self.dec_name].sum(axis=1).tolist()
            res2 = tmp2[self.val_name].sum(axis=1).tolist()
            tmp_cor = round(np.corrcoef(res1,res2)[0][1],3)
        
            #plt.scatter(res1,res2,label=d+" : "+str(tmp_cor),alpha=1.0,s=self.plot_size) # inner correlation
            plt.scatter(res1,res2,label=d,alpha=1.0,s=self.plot_size)
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
        plt.legend(loc="upper left", bbox_to_anchor=(1, 1),shadow=True,fontsize=13)
        plt.xlabel(self.xlabel,fontsize=self.label_size)
        plt.ylabel(self.ylabel,fontsize=self.label_size)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().yaxis.set_ticks_position('left')
        plt.gca().xaxis.set_ticks_position('bottom')
        ax.set_axisbelow(True)
        ax.grid(color="#ababab",linewidth=0.5)
        plt.title(title,fontsize=15)
        plt.show()

        return performance,total_x,total_y
    
    def overlap_singles(self,evalxy, title_list=['Naive B','Naive CD4 T','CD8 T','NK','Monocytes']):
        total_x = []
        for t in evalxy[0]:
            total_x.extend(t)
        total_y = []
        for t in evalxy[1]:
            total_y.extend(t)
        
        total_cor, pvalue = stats.pearsonr(total_x,total_y) # correlation and pvalue
        total_cor = round(total_cor,4)
        if pvalue < 0.01:
            pvalue = '{:.2e}'.format(pvalue)
        else:
            pvalue = round(pvalue,3)
        rmse = round(np.sqrt(mean_squared_error(total_x, total_y)),4)
        performance = {'R':total_cor,'P':pvalue,'RMSE':rmse}

        x_min = min(min(total_x),min(total_y))
        x_max = max(max(total_x),max(total_y))

        fig,ax = plt.subplots(figsize=self.figsize,dpi=self.dpi)
        for idx in range(len(evalxy[0])):
            res1 = evalxy[0][idx]
            res2 = evalxy[1][idx]
            cell = title_list[idx]

            plt.scatter(res1,res2,alpha=0.8,s=60,label=cell)
            plt.plot([x_min,x_max],[x_min,x_max],linewidth=2,color='black',linestyle='dashed',zorder=-1)

        plt.text(1.0,0.15,'R = {}'.format(str(round(total_cor,3))), transform=ax.transAxes, fontsize=15)
        plt.text(1.0,0.10,'P = {}'.format(str(pvalue)), transform=ax.transAxes, fontsize=15)
        plt.text(1.0,0.05,'RMSE = {}'.format(str(round(rmse,3))), transform=ax.transAxes, fontsize=15)
        #plt.legend(shadow=True)
        plt.xlabel('Estimated Proportion',fontsize=12)
        plt.ylabel('True Proportion',fontsize=12)
        plt.xticks(fontsize=self.tick_size)
        plt.yticks(fontsize=self.tick_size)

        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().yaxis.set_ticks_position('left')
        plt.gca().xaxis.set_ticks_position('bottom')
        ax.set_axisbelow(True)
        ax.grid(color="#ababab",linewidth=0.5)
        #plt.title(title,fontsize=12)
        plt.legend(shadow=True,bbox_to_anchor=(1.0, 1), loc='upper left')
        plt.show()

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
    
def plot_multi(data=[[11,50,37,202,7],[47,19,195,117,74],[136,69,33,47],[100,12,25,139,89]],names=["+PBS","+Nefopam","+Ketoprofen","+Cefotaxime"],value="ALT (U/I)",title="",grey=True,dpi=100,figsize=(12,6),lw=1,capthick=1,capsize=5):
    sns.set()
    sns.set_style('whitegrid')
    if grey:
        sns.set_palette('gist_yarg')
        
    fig,ax = plt.subplots(figsize=figsize,dpi=dpi)
    
    df = pd.DataFrame()
    for i in range(len(data)):
        tmp_df = pd.DataFrame({names[i]:data[i]})
        df = pd.concat([df,tmp_df],axis=1)
    error_bar_set = dict(lw=lw,capthick=capthick,capsize=capsize)
    if grey:
        ax.bar([i for i in range(len(data))],df.mean(),yerr=df.std(),tick_label=df.columns,error_kw=error_bar_set)
    else:
        ax.bar([i for i in range(len(data))],df.mean(),yerr=df.std(),tick_label=df.columns,error_kw=error_bar_set)
    # jitter plot
    df_melt = pd.melt(df)
    sns.stripplot(x='variable', y='value', data=df_melt, jitter=True, color='black', ax=ax, size=3)
        
    ax.set_xlabel('')
    ax.set_ylabel(value)
    plt.title(title)
    plt.xticks(rotation=60)
    plt.show()

def plot_rader(data=[[0.3821, 0.6394, 0.8317, 0.7524],[0.4908, 0.7077, 0.8479, 0.7802]],labels=['Neutrophils', 'Monocytes', 'NK', 'Kupffer'],conditions=['w/o addnl. topic','w/ addnl. topic'],title='APAP Treatment'):
    # preprocessing
    dft = pd.DataFrame(data,index=conditions)

    # Split the circle into even parts and save the angles
    # so we know where to put each axis.
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()

    # The plot is a circle, so we need to "complete the loop"
    # and append the start value to the end.
    angles += angles[:1]

    # ax = plt.subplot(polar=True)
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True), dpi=100)

    # Helper function to plot each car on the radar chart.
    def add_to_radar(name, color):
        values = dft.loc[name].tolist()
        values += values[:1]
        ax.plot(angles, values, color=color, linewidth=2, label=name)
        ax.fill(angles, values, color=color, alpha=0.25)
    # Add each car to the chart.
    add_to_radar(conditions[0], '#429bf4')
    add_to_radar(conditions[1], '#ec6e95')

    # Fix axis to go in the right order and start at 12 o'clock.
    ax.set_theta_offset(np.pi/4)
    ax.set_theta_direction(-1)

    # Draw axis lines for each angle and label.
    ax.set_thetagrids(np.degrees(angles)[0:len(labels)], labels)

    # Go through labels and adjust alignment based on where
    # it is in the circle.
    for label, angle in zip(ax.get_xticklabels(), angles):
        if angle in (0, np.pi):
            label.set_horizontalalignment('left')
        elif 0 < angle < np.pi:
            label.set_horizontalalignment('left')
        else:
            label.set_horizontalalignment('right')

    # Ensure radar range
    #ax.set_ylim(0, 0.9)
    ax.set_rlabel_position(180 / len(labels))
    ax.tick_params(colors='#222222')
    ax.tick_params(axis='y', labelsize=8)
    ax.grid(color='#AAAAAA')
    ax.spines['polar'].set_color('#222222')
    ax.set_facecolor('#FAFAFA')
    ax.set_title(title, y=1.02, fontsize=15)
    ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
    plt.show()
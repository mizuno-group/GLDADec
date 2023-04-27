#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 27 13:19:33 2022

@author: docker
"""
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from scipy import stats
import matplotlib.pyplot as plt

def standardz_sample(x):
    pop_mean = x.mean(axis=0)
    pop_std = x.std(axis=0)
    df = (x - pop_mean).divide(pop_std)
    df = df.replace(np.inf,np.nan)
    df = df.replace(-np.inf,np.nan)
    df = df.dropna()
    #print('standardz population control')
    return df

def plot_simple_corr(deconv_df,val_df,dec_name=['B cells naive'],val_name=['Naive B'],do_plot=True,sep=True,do_print=False,dpi=300):
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
    total_cor = round(np.corrcoef(deconv_df[dec_name].sum(axis=1).tolist(),val_df[val_name].sum(axis=1).tolist())[0][1],4)
    
    if do_print:
        print(str(dec_name)+" vs "+str(val_name))
        print(total_cor)
    
    res1 = deconv_df[dec_name].sum(axis=1).tolist()
    res2 = val_df[val_name].sum(axis=1).tolist()
    tmp_cor = round(np.corrcoef(res1,res2)[0][1],3)
    label = dec_name[0]+" : "+str(tmp_cor)
    
    x_min = min(min(res1),min(res2))
    x_max = max(max(res1),max(res2))
    
    if do_plot:
        fig,ax = plt.subplots(figsize=(6,6),dpi=dpi)
        plt.scatter(res1,res2,label=label,alpha=1.0)
        plt.plot([x_min,x_max],[x_min,x_max],linewidth=2,color='black',linestyle='dashed',zorder=-1)
        plt.text(0.3,0.05,'Cor = {}'.format(str(round(total_cor,3))), transform=ax.transAxes, fontsize=15)
        
        plt.legend(shadow=True)
        plt.xlabel("Deconvolution estimated value")
        plt.ylabel("FACS value")
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().yaxis.set_ticks_position('left')
        plt.gca().xaxis.set_ticks_position('bottom')
        ax.set_axisbelow(True)
        ax.grid(color="#ababab",linewidth=0.5)
        plt.title(str(dec_name)+" vs "+str(val_name))
        plt.show()
    else:
        pass
    return total_cor,res1,res2,label

def plot_group_corr(deconv_df,val_df,dec_name=["CD4","CD8"],val_name=["abT"],sort_index=[],
                    do_plot=True,sep=True,title=None,do_print=False,dpi=300):
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
    total_cor = round(np.corrcoef(total_x,total_y)[0][1],4)
    
    if do_print:
        print(str(dec_name)+" vs "+str(val_name))
        print(total_cor)
    
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
        
            plt.scatter(res1,res2,label=d+" : "+str(tmp_cor),alpha=1.0)
            
            xmin = min(min(res1),min(res2))
            if xmin < x_min:
                x_min = xmin
            xmax = max(max(res1),max(res2))
            if xmax > x_max:
                x_max = xmax
        
        plt.plot([x_min,x_max],[x_min,x_max],linewidth=2,color='black',linestyle='dashed',zorder=-1)
        
        plt.text(0.3,0.05,'Cor = {}'.format(str(round(total_cor,3))), transform=ax.transAxes, fontsize=15)
        
        plt.legend(shadow=True)
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

def plot_cor_var(total_res,val_df,dec_name=['B cells naive'],val_name=['Naive B'],
                 do_plot=True,sep=True,do_print=False,plot_size=100,dpi=300):
    cell_summary = pd.DataFrame()
    for tmp_df in total_res:
        try:
            cell_df = pd.DataFrame(tmp_df[dec_name].sum(axis=1))
            cell_summary = pd.concat([cell_summary,cell_df],axis=1)
        except:
            pass
    cell_avg = pd.DataFrame(cell_summary.mean(axis=1))
    cell_var = pd.DataFrame(cell_summary.var(axis=1))
        
    # align sample order
    samples = val_df.index.tolist()
    cell_avg = cell_avg.loc[samples]
    cell_var = cell_var.loc[samples]
    facs_v = pd.DataFrame(val_df[val_name].sum(axis=1))

    # normalization
    z_res = standardz_sample(cell_avg)
    z_ref = standardz_sample(facs_v)
    
    total_cor = round(np.corrcoef(z_res[0].tolist(),z_ref[0].tolist())[0][1],4)
    label = dec_name[0]+" : "+str(round(total_cor,3))
    
    x_min = min(min(z_res[0]),min(z_ref[0]))
    x_max = max(max(z_res[0]),max(z_ref[0]))

    
    if do_plot:
        fig,ax = plt.subplots(figsize=(6,6),dpi=dpi)
        cell_var_avg = cell_var.mean()
        #cell_var_norm = cell_var * plot_size / cell_var_avg
        cell_var_norm = plot_size / cell_var
        
        plt.scatter(z_res[0].tolist(),z_ref[0].tolist(),s=cell_var_norm[0].tolist(),label=label,alpha=0.9)
        plt.plot([x_min,x_max],[x_min,x_max],linewidth=2,color='black',linestyle='dashed',zorder=-1)
        plt.text(0.3,0.05,'Cor = {}'.format(str(round(total_cor,3))), transform=ax.transAxes, fontsize=15)
    
        plt.legend(shadow=True)
        plt.xlabel("Deconvolution estimated value")
        plt.ylabel("FACS value")
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().yaxis.set_ticks_position('left')
        plt.gca().xaxis.set_ticks_position('bottom')
        ax.set_axisbelow(True)
        ax.grid(color="#ababab",linewidth=0.5)
        plt.title(str(dec_name)+" vs "+str(val_name))
        plt.show()
    else:
        pass
    
    return total_cor,z_res[0].tolist(),z_ref[0].tolist(),label

def plot_cor_publication(total_res,val_df,dec_name=['B cells naive'],val_name=['Naive B'],
                 do_plot=True,sep=True,do_print=False,plot_size=100,dpi=100,color='tab:blue'):
    cell_summary = pd.DataFrame()
    for tmp_df in total_res:
        try:
            cell_df = pd.DataFrame(tmp_df[dec_name].sum(axis=1))
            cell_summary = pd.concat([cell_summary,cell_df],axis=1)
        except:
            pass
    cell_avg = pd.DataFrame(cell_summary.mean(axis=1))
    cell_var = pd.DataFrame(cell_summary.var(axis=1))
        
    # align sample order
    samples = val_df.index.tolist()
    cell_avg = cell_avg.loc[samples]
    cell_var = cell_var.loc[samples]
    facs_v = pd.DataFrame(val_df[val_name].sum(axis=1))

    # normalization
    z_res = standardz_sample(cell_avg)
    z_ref = standardz_sample(facs_v)
    
    total_cor = round(np.corrcoef(z_res[0].tolist(),z_ref[0].tolist())[0][1],4)
    label = dec_name[0]+" : "+str(round(total_cor,3))
    
    x_min = min(min(z_res[0]),min(z_ref[0]))
    x_max = max(max(z_res[0]),max(z_ref[0]))

    
    if do_plot:
        fontsize=15
        fig,ax = plt.subplots(figsize=(6,6),dpi=dpi)
        cell_var_avg = cell_var.mean()
        #cell_var_norm = cell_var * plot_size / cell_var_avg
        cell_var_norm = plot_size / cell_var
        
        plt.scatter(z_res[0].tolist(),z_ref[0].tolist(),s=cell_var_norm[0].tolist(),label=label,alpha=0.9,color=color)
        plt.plot([x_min,x_max],[x_min,x_max],linewidth=2,color='black',linestyle='dashed',zorder=-1)
        plt.text(0.6,0.05,'Cor = {}'.format(str(round(total_cor,3))), transform=ax.transAxes, fontsize=15)
    
        #plt.legend(shadow=True)
        plt.xlabel("Deconvolution value",fontsize=fontsize)
        plt.ylabel("Flow cytometry value",fontsize=fontsize)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().yaxis.set_ticks_position('left')
        plt.gca().xaxis.set_ticks_position('bottom')
        ax.set_axisbelow(True)
        ax.grid(color="#ababab",linewidth=0.5)
        plt.title(str(dec_name[0]),fontsize=fontsize)
        plt.show()
    else:
        pass
    
    return total_cor,z_res[0].tolist(),z_ref[0].tolist(),label

def plot_cor_error(total_res,val_df,dec_name=['B cells naive'],val_name=['Naive B'],
                   do_plot=True,sep=True,do_print=False,plot_all=False,dpi=300):
    cell_summary = pd.DataFrame()
    for tmp_df in total_res:
        cell_df = pd.DataFrame(tmp_df[dec_name].sum(axis=1))
        cell_summary = pd.concat([cell_summary,cell_df],axis=1)
    
    cell_avg = pd.DataFrame(cell_summary.mean(axis=1))
    cell_z_summary = standardz_sample(cell_summary)
    cell_std = pd.DataFrame(cell_z_summary.std(axis=1))
        
    # align sample order
    samples = val_df.index.tolist()
    facs_v = pd.DataFrame(val_df[val_name].sum(axis=1))

    # normalization
    z_res = standardz_sample(cell_avg)
    z_ref = standardz_sample(facs_v)

    total_cor = round(np.corrcoef(z_res[0].tolist(),z_ref[0].tolist())[0][1],4)
    label = dec_name[0]+" : "+str(round(total_cor,3))
    
    x_min = min(min(z_res[0]),min(z_ref[0]))
    x_max = max(max(z_res[0]),max(z_ref[0]))
        
    # plot
    fig,ax = plt.subplots(figsize=(6,6),dpi=dpi)
    for j,s in enumerate(samples):
        x = cell_z_summary.loc[s].tolist()
        y = z_ref.loc[s].tolist()*len(x)
        plt.errorbar(np.mean(x),y=z_ref.loc[s].tolist(),xerr=cell_std.loc[s].tolist(),
                     capsize=5,marker='_',fmt='o',ms=8,mew=1,zorder=-1,alpha=0.8)
        if plot_all:
            plt.scatter(x,y,alpha=0.8,s=30,linewidth=0.4,edgecolor='k')
        else:
            plt.scatter(np.mean(x),y=z_ref.loc[s].tolist(),alpha=0.8,s=100,linewidth=0.4,edgecolor='k')

    plt.plot([x_min,x_max],[x_min,x_max],linewidth=2,color='black',linestyle='dashed',zorder=-1)
    plt.text(0.3,0.05,'Cor = {}'.format(str(round(total_cor,3))), transform=ax.transAxes, fontsize=15)

    plt.xlabel("Deconvolution estimated value")
    plt.ylabel("FACS value")
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().yaxis.set_ticks_position('left')
    plt.gca().xaxis.set_ticks_position('bottom')
    ax.set_axisbelow(True)
    ax.grid(color="#ababab",linewidth=0.5)
    plt.title(str(dec_name)+" vs "+str(val_name))
    plt.show()
    
    return total_cor,z_res[0].tolist(),z_ref[0].tolist(),label

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


def plot_multi(data=[[11,50,37,202,7],[47,19,195,117,74],[136,69,33,47],[100,12,25,139,89]],names = ["+PBS","+Nefopam","+Ketoprofen","+Cefotaxime"],value="ALT (U/I)",title="",grey=True, dpi=100):
    sns.set()
    sns.set_style('whitegrid')
    if grey:
        sns.set_palette('gist_yarg')
        
    fig,ax = plt.subplots(figsize=(12,6),dpi=dpi)
    
    df = pd.DataFrame()
    for i in range(len(data)):
        tmp_df = pd.DataFrame({names[i]:data[i]})
        df = pd.concat([df,tmp_df],axis=1)
    error_bar_set = dict(lw=1,capthick=1,capsize=5)
    if grey:
        ax.bar([i for i in range(len(data))],df.mean(),yerr=df.std(),tick_label=df.columns,error_kw=error_bar_set)
    else:
        ax.bar([i for i in range(len(data))],df.mean(),yerr=df.std(),tick_label=df.columns,error_kw=error_bar_set)
    # jitter plot
    df_melt = pd.melt(df)
    sns.stripplot(x='variable', y='value', data=df_melt, jitter=True, color='black', ax = ax, size=3)
        
    ax.set_xlabel('')
    ax.set_ylabel(value)
    plt.title(title)
    plt.xticks(rotation=60)
    plt.show()

def group_ttest(df,target_samples=['Ctrl','APAP']):
    samples = df.columns.tolist()
    ctrl_samples = []
    treat_samples = []
    for t in samples:
        if t.split('_')[0] == target_samples[0]:
            ctrl_samples.append(t)
        elif t.split('_')[0] == target_samples[1]:
            treat_samples.append(t)
        else:
            pass

    ctrl_df = df[ctrl_samples]
    treat_df = df[treat_samples]

    # Welch t-test
    whole_genes = df.index.tolist()
    stat_res = []
    p_res = []
    for gene in tqdm(whole_genes):
        ctrl_v = np.array(ctrl_df.loc[gene])
        treat_v = np.array(treat_df.loc[gene])
        stat, p = stats.ttest_ind(ctrl_v, treat_v, equal_var=False)
        stat_res.append(stat)
        p_res.append(p)

    res_df = pd.DataFrame({'statistic':stat_res, 'pvalue':p_res},index=whole_genes)
    return res_df


# -*- coding: utf-8 -*-
"""
Created on 2023-08-26 (Sat) 13:46:23

@author: I.Azuma
"""
# %%
import pandas as pd
from glob import glob
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.collections import EllipseCollection

# %%
def heatmap_eval(baseline_dir = '/workspace/github/GLDADec/baselines_eval/_performance_summary/GSE65133',
                  methods = ['Proposed','FARDEEP','EPIC','CIBERSORT','DCQ','NNLS','RLR','ElasticNet'],
                  cells = ['Naive B', 'Memory B', 'Naive CD4 T', 'Memory CD4 T', 'CD8 T', 'NK', 'Monocytes', 'Gamma delta T']):
    cor_data = []
    rmse_data = []
    for method in methods:
        path = glob(baseline_dir+'/{}*.pkl'.format(method))

        if len(path) != 1:
            print(path)
            raise ValueError('something is wrong')
        res = pd.read_pickle(path[0])

        cor_list = []
        rmse_list = []
        for cell in cells:
            cell_res = res.get(cell)
            cor_list.append(cell_res[0][1])
            rmse_list.append(cell_res[2][1])
        cor_data.append(cor_list)
        rmse_data.append(rmse_list)

    # plot bar
    data = cor_data
    hatch_list = ['/', '|', '-', '+', 'x', 'o', 'O', '.', '*']
    fig,ax = plt.subplots(figsize=(15,8),dpi=300)
    x = [k*len(data) for k in range(len(cells))]
    for i in range(len(data)):
        v = data[i]
        x2 = [t+i*0.8 for t in x]
        plt.bar(x2,v,label=methods[i],width=0.7,hatch=hatch_list[i]*2)
        #plt.bar(x2,v,label=methods[i],width=0.7)
    plt.xticks([t+2.4 for t in x],cells,fontsize=18,rotation=90)
    plt.yticks(fontsize=18)
    plt.ylabel('Correlation',fontsize=18)
    plt.legend(loc='upper center',shadow=True,fontsize=13,bbox_to_anchor=(0.5,1.1),ncol=len(methods))
    #plt.legend(loc='best',fontsize=13,shadow=True)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().yaxis.set_ticks_position('left')
    plt.gca().xaxis.set_ticks_position('bottom')
    ax.set_axisbelow(True)
    ax.grid(color="#ababab",linewidth=1.0, axis='y')
    plt.show()

    # display heatmap
    cor_df = pd.DataFrame(data)
    sns.heatmap(cor_df,annot=True,linewidths=0.5,cmap='cividis',fmt='.2f')
    plt.xticks([i+0.5 for i in range(len(cells))],cells,rotation=90)
    plt.yticks([i+0.5 for i in range(len(methods))],methods,rotation=0)
    plt.show()


# %%
def cor_rmse_eval(baseline_dir = '/workspace/github/GLDADec/baselines_eval/_performance_summary/GSE65133',
                  methods = ['Proposed','FARDEEP','EPIC','CIBERSORT','DCQ','NNLS','RLR','ElasticNet'],
                  cells = ['Naive B', 'Memory B', 'Naive CD4 T', 'Memory CD4 T', 'CD8 T', 'NK', 'Monocytes', 'Gamma delta T']):
    cor_data = []
    rmse_data = []
    for method in methods:
        path = glob(baseline_dir+'/{}*.pkl'.format(method))

        if len(path) != 1:
            print(path)
            raise ValueError('something is wrong')
        res = pd.read_pickle(path[0])

        cor_list = []
        rmse_list = []
        for cell in cells:
            cell_res = res.get(cell)
            cor_list.append(cell_res[0][1])
            rmse_list.append(cell_res[2][1])
        cor_data.append(cor_list)
        rmse_data.append(rmse_list)

    M = np.array(cor_data)
    M2 = np.array(rmse_data)
    fig, ax = plt.subplots(1, 1, subplot_kw={'aspect':'equal'})
    ax.set_xlim(-0.5, M.shape[1] - 0.5)
    ax.set_ylim(-0.5, M.shape[0] - 0.5)
    # xy locations of each ellipse center
    xy = np.indices(M.shape)[::-1].reshape(2, -1).T
    new_xy = [[t[0],7-t[1]] for t in xy]

    h = M.ravel()
    a = 45 * np.sign(M).ravel()
    ec = EllipseCollection(widths=h, heights=h, angles=a, units='x', offsets=new_xy,
                            transOffset=ax.transData, array=M2.ravel())
    ax.add_collection(ec)
    cb = fig.colorbar(ec)
    cb.set_label('RMSE')
    ax.margins(0.1)
    ax.set_xticks(np.arange(M.shape[1]))
    ax.set_xticklabels(cells, rotation=90)
    ax.set_yticks(np.arange(M.shape[0]))
    ax.set_yticklabels(reversed(methods))
    plt.show()

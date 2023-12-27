# -*- coding: utf-8 -*-
"""
Created on 2023-06-01 (Thu) 17:27:37

Prep classical marker genes
GSE176082
- https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9109816/

@author: I.Azuma
"""
#%%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#%% Figure 4(A)
# neutrophils
neu_candi = ['S100A9','S100A8','CAMP','NP4','NGP']

# monocytes
mon_candi = ['FCNB','LYZ2','LY6C']

# macrophage
mac_candi = ['C1QA','C1QC','C1QB','VCAM1']

# T cells
t_candi = ['CD3G','CD3D']

# NK
# nk_candi = ['NKG7','KLRB1A','KLRD1'] KLRB1A is not contained
nk_candi = ['NKG7','KLRB1','KLRD1']

# B
b_candi = ['MS4A1','MZB1','PDIA4','PDIA6','TMEM97']

# erythroid
#ery_candi = ['HBA','HBB']
ery_candi = ['HBA-A1','HBA-A2','HBA-A3','HBB']

# create dict
gse176082_fig4_dic = dict(zip(['Neu','Mon','Mac','T','NK','B','Ery'],[neu_candi,mon_candi,mac_candi,t_candi,nk_candi,b_candi,ery_candi]))

# save
#pd.to_pickle(gse176082_fig4_dic,'/workspace/github/GLDADec/data/marker/how_processed/marker_prep_rat_liver.py')

# %%
# -*- coding: utf-8 -*-
"""
Created on 2023-12-27 (Wed) 14:14:30

Prep DEGs derived marker genes.

@author: I.Azuma
"""
import numpy as np
import pandas as pd
import seaborn as sns
import sys
sys.path.append('/workspace/github/LiverDeconv')
import liver_deconv

# %% APAP
# up to 20
rat_df = pd.read_csv('/workspace/github/GLDADec/data/expression/rat/Morita_rat/vs_APAP/rat_dili_APAP_expression.csv',index_col=0)
ref_df = pd.read_csv('/workspace/github/LiverDeconv/Data/processed/ref_13types.csv',index_col=0)

dat = liver_deconv.LiverDeconv()
dat.set_data(df_mix=rat_df, df_all=ref_df)
dat.pre_processing(do_ann=False,ann_df=None,do_log2=True,do_quantile=False,do_trimming=False,do_drop=True)
dat.narrow_intersec()
dat.create_ref(sep="_",number=20,limit_CV=10,limit_FC=1.5,log2=False,verbose=True,do_plot=True)
deg_dic = dat.deg_dic

pd.to_pickle(deg_dic,'/workspace/github/GLDADec/data/marker/mouse_LM6_DEGs.pkl')
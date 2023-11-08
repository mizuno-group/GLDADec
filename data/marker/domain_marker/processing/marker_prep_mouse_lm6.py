# -*- coding: utf-8 -*-
"""
Created on 2023-07-31 (Mon) 19:06:12

marker prep for mouse LM6

@author: I.Azuma
"""
# %%
import numpy as np
import pandas as pd
import seaborn as sns
import sys
sys.path.append('/workspace/github/LiverDeconv')
import liver_deconv

# %%
"""
Mouse LM13 derived DEGs for rat analysis (APAP)
"""
# up to 10
rat_df = pd.read_csv('/workspace/github/GLDADec/data/expression/rat/Morita_rat/vs_APAP/rat_dili_APAP_expression.csv',index_col=0)
ref_df = pd.read_csv('/workspace/github/LiverDeconv/Data/processed/ref_13types.csv',index_col=0)

# sample selection
target_cells = ['B','NK','CD4','CD8','Monocyte','Neutrophil']
use_samples = []
for t in ref_df.columns.tolist():
    if t.split('_')[0] in target_cells:
        use_samples.append(t)
    else:
        pass

ref_target = ref_df[sorted(use_samples)]

dat = liver_deconv.LiverDeconv()
dat.set_data(df_mix=rat_df, df_all=ref_target)
dat.pre_processing(do_ann=False,ann_df=None,do_log2=True,do_quantile=False,do_trimming=False,do_drop=True)
dat.narrow_intersec()
dat.create_ref(sep="_",number=10,limit_CV=10,limit_FC=1.5,log2=False,verbose=True,do_plot=True)
deg_dic = dat.deg_dic

pd.to_pickle(deg_dic,'/workspace/github/GLDADec/data/expression/rat/marker/MouseLM6_DEGs/lm6_10.pkl')
# %%
# up to 20
rat_df = pd.read_csv('/workspace/github/GLDADec/data/expression/rat/Morita_rat/vs_APAP/rat_dili_APAP_expression.csv',index_col=0)
ref_df = pd.read_csv('/workspace/github/LiverDeconv/Data/processed/ref_13types.csv',index_col=0)

# sample selection
target_cells = ['B','NK','CD4','CD8','Monocyte','Neutrophil']
use_samples = []
for t in ref_df.columns.tolist():
    if t.split('_')[0] in target_cells:
        use_samples.append(t)
    else:
        pass

ref_target = ref_df[sorted(use_samples)]

dat = liver_deconv.LiverDeconv()
dat.set_data(df_mix=rat_df, df_all=ref_target)
dat.pre_processing(do_ann=False,ann_df=None,do_log2=True,do_quantile=False,do_trimming=False,do_drop=True)
dat.narrow_intersec()
dat.create_ref(sep="_",number=20,limit_CV=10,limit_FC=1.5,log2=False,verbose=True,do_plot=True)
deg_dic = dat.deg_dic

pd.to_pickle(deg_dic,'/workspace/github/GLDADec/data/expression/rat/marker/MouseLM6_DEGs/lm6_20.pkl')
# %%

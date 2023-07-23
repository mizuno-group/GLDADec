# -*- coding: utf-8 -*-
"""
Created on 2023-05-30 (Tue) 15:37:27

marker prep for rat

@author: I.Azuma
"""
#%%
import numpy as np
import pandas as pd
import seaborn as sns
import sys
sys.path.append('/workspace/github/LiverDeconv')
import liver_deconv

#%% up to 50
ref_df = pd.read_csv('/workspace/github/GLDADec/data/rat/Morita_rat/df_ref.csv',index_col=0)

dat = liver_deconv.LiverDeconv()
dat.set_data(df_mix=ref_df, df_all=ref_df)
dat.pre_processing(do_ann=False,ann_df=None,do_log2=True,do_quantile=False,do_trimming=False,do_drop=True)
dat.narrow_intersec()
dat.create_ref(sep="_",number=50,limit_CV=10,limit_FC=1.5,log2=False,verbose=True,do_plot=True)
deg_dic = dat.deg_dic

pd.to_pickle(deg_dic,'/workspace/github/GLDADec/data/rat/marker/DEGs/rnaseq_50.pkl')

#%% up to 10
ref_df = pd.read_csv('/workspace/github/GLDADec/data/rat/Morita_rat/df_ref.csv',index_col=0)

dat = liver_deconv.LiverDeconv()
dat.set_data(df_mix=ref_df, df_all=ref_df)
dat.pre_processing(do_ann=False,ann_df=None,do_log2=True,do_quantile=False,do_trimming=False,do_drop=True)
dat.narrow_intersec()
dat.create_ref(sep="_",number=10,limit_CV=10,limit_FC=1.5,log2=False,verbose=True,do_plot=True)

final_ref = dat.final_ref
deg_dic = dat.deg_dic
pd.to_pickle(deg_dic,'/workspace/github/GLDADec/data/rat/marker/DEGs/rnaseq_10.pkl')

#%% up to 100
ref_df = pd.read_csv('/workspace/github/GLDADec/data/rat/Morita_rat/df_ref.csv',index_col=0)

dat = liver_deconv.LiverDeconv()
dat.set_data(df_mix=ref_df, df_all=ref_df)
dat.pre_processing(do_ann=False,ann_df=None,do_log2=True,do_quantile=False,do_trimming=False,do_drop=True)
dat.narrow_intersec()
dat.create_ref(sep="_",number=100,limit_CV=10,limit_FC=1.2,log2=False,verbose=True,do_plot=True)

final_ref = dat.final_ref
deg_dic = dat.deg_dic
pd.to_pickle(deg_dic,'/workspace/github/GLDADec/data/rat/marker/DEGs/rnaseq_100.pkl')


#%% 230531 domain
# B cell
bcandi = ['BCR','CD79A','CD79B'] # https://www.bdj.co.jp/pdf/55-28_00-6081-69A1.pdf

# CD4T cell
cd4candi = ['CD4']

# CD8T cell
cd8candi = ['CD8A','CD8B']

# Monocyte
moncandi = ['CD68','CD18'] # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3644679/, https://www.deepl.com/ja/translator

# NK cell
nkcandi = ['KLRB1','KLRB1B','KLRB1C'] # NKR-P1A (https://www.bdj.co.jp/pdf/55-28_00-6081-69A1.pdf)

# Neutrophils
neucandi = ['IL1R2','S100A8','S100A9'] # https://pubmed.ncbi.nlm.nih.gov/23817563/,https://www.frontiersin.org/articles/10.3389/fimmu.2018.01298/full

domain_dic = dict(zip(['B', 'CD4T', 'CD8T', 'Mon', 'NK', 'Neu'],[bcandi,cd4candi,cd8candi,moncandi,nkcandi,neucandi]))
pd.to_pickle(domain_dic,'/workspace/github/GLDADec/data/rat/marker/domain/domain_v1_dic.pkl')

#%%
ref_df = pd.read_csv('/workspace/github/GLDADec/data/rat/Morita_rat/df_ref.csv',index_col=0)
genes = [t.upper() for t in ref_df.index.tolist()]
ref_df.index = genes
for g in genes:
    if 'CD8' in str(g):
        print(g)
    else:
        pass

#%%
tmp_df = ref_df.loc[['CD8A','CD8B']]
sns.heatmap(tmp_df)

df_CV = np.std(ref_df.T) / np.mean(ref_df.T)
df_CV.index = ref_df.index
df_CV = df_CV.replace(np.inf,np.nan)
df_CV = df_CV.replace(-np.inf,np.nan)
df_CV = df_CV.dropna()
df_CV=pd.DataFrame(df_CV)

tmp_df = df_CV.loc[['CD8A','CD8B']]

deg_dic = pd.read_pickle('/workspace/github/GLDADec/data/rat/marker/DEGs/rnaseq_50.pkl')
deg_dic.get('CD8T')
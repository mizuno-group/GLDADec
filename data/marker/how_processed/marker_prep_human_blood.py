# -*- coding: utf-8 -*-
"""
Created on 2023-06-15 (Thu) 15:29:39

Human blood related marker definition with domain knowledge.

@author: I.Azuma
"""
#%%
import pandas as pd

# %%
mon_marker = ['CCR2', 'CD14', 'CD68', 'CD86', 'FCGR1A', 'ITGAM', 'KIT']
cd14mon_marker = ['CCR2', 'CD14', 'CD68', 'CD86', 'FCGR1A', 'ITGAM', 'KIT']
cd16mon_marker = ['CCR2', 'FCGR3A', 'CD68', 'CD86', 'FCGR1A', 'ITGAM', 'KIT']

nk_marker = ['CX3CR1', 'IL2RB','PTPRC', 'SELL','FCGR3A','FCGR3B']
nk_marker2 = ['CX3CR1', 'IL2RB', 'PTPRC', 'SELL']

bn_marker = ['BCL7A', 'FCER2', 'IGHD', 'IGHM', 'PAX5', 'TCL1A']
bm_marker = ['AIM2', 'CR2', 'JCHAIN']

cd4n_marker = ['CD28', 'EEF1B2', 'FHIT', 'GIMAP5', 'GIMAP8', 'PRKCA', 'RPS5', 'RSL1D1', 'SATB1', 'SLC40A1', 'SVIL', 'TESPA1', 'TSHZ2']
cd4m_marker = ['CCR6', 'DPP4']

cd8_marker = ['CD8A', 'CD8B']

gd_marker = ['S100B', 'TRGV9', 'TRGV1', 'CCL5', 'STMN1', 'TRGJ2', 'HMGB2', 'TRGJP2', 'TRGV3', 'TRGV11', 'H2AFZ', 'NUSAP1', 'TROAP', 'TUBB']

neu_marker = ['CD14', 'CD86', 'CXCR1', 'CXCR2', 'FCGR3A', 'ITGAM', 'PTPRC', 'SELL']

dc_marker = ['CD86', 'CD80', 'CD83', 'IL3RA', 'NRP1', 'CLEC4C', 'JCHAIN', 'LILRA4', 'MZB1', 'IL5RA', 'MS4A3', 'FCER1A', 'HPGD', 'CST3', 'PLD4', 'ISG15', 'UBC', 'CLEC10A', 'IRF8', 'IFITM3', 'IFI6', 'CYBA', 'CD1C', 'HSPA5', 'IRF7', 'CLEC12A', 'LILRB4', 'CD1E', 'THBD']

treg_marker = ['CCR4', 'CD27', 'CD4', 'CTLA4', 'FOXP3', 'ICOS', 'IKZF2', 'IL2RA']

k = ['Monocytes','CD14+ Mon','CD16+ Mon','NK cells','NK cells 2','B cells naive','B cells memory','T cells CD4 naive','T cells CD4 memory','T cells CD8','T cells gamma delta','Neutrophils','Dendritic cells','Treg']
v = [mon_marker,cd14mon_marker,cd16mon_marker,nk_marker,nk_marker2,bn_marker,bm_marker,cd4n_marker,cd4m_marker,cd8_marker,gd_marker,neu_marker,dc_marker,treg_marker]
domain_dic = dict(zip(k,v))

# pd.to_pickle(domain_dic,'/workspace/github/GLDADec/data/marker/human_blood_domain.pkl')
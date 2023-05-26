# -*- coding: utf-8 -*-
"""
Created on 2023-05-24 (Wed) 08:44:39

marker prep for human blood related cells

@author: I.Azuma
"""
#%%
import pandas as pd
import codecs
import collections
import itertools

Base_dir = '/workspace/github/GLDADec' # cloning repository

#%% Blood all related
# load total data
with codecs.open(Base_dir + '/data/domain_info/CellMarker/Cell_marker_All.csv', "r", "Shift-JIS", "ignore") as file:
    total_ref = pd.read_table(file, delimiter=",")

# human
human_ref = total_ref[total_ref["species"].isin(["Human"])] # 60877
target_tissues = ['Blood'] # blood
target_ref = human_ref[human_ref['tissue_class'].isin(target_tissues)]
target_ref = target_ref[target_ref['cell_type']=='Normal cell']

cell_names = target_ref['cell_name'].unique().tolist() # 342

# integrate
name_res = []
symbol_res = []
for cell in cell_names:
    tmp_df = target_ref[target_ref['cell_name']==cell]
    symbols = tmp_df['Symbol'].dropna().unique().tolist()
    if len(symbols) > 0:
        name_res.append(cell)
        symbol_res.append(symbols)
    else:
        pass
blood_dic = dict(zip(name_res,symbol_res)) # 328 cell types
pd.to_pickle(blood_dic,'/workspace/github/GLDADec/data/GSE65133/230524/human_blood_328_dic.pkl')

#%%
# load total data
with codecs.open(Base_dir + '/data/domain_info/CellMarker/Cell_marker_All.csv', "r", "Shift-JIS", "ignore") as file:
    total_ref = pd.read_table(file, delimiter=",")

# human
human_ref = total_ref[total_ref["species"].isin(["Human"])] # 60877
target_tissues = ['Peripheral blood'] # blood
target_ref = human_ref[human_ref['tissue_type'].isin(target_tissues)]
target_ref = target_ref[target_ref['cell_type']=='Normal cell']

cell_names = target_ref['cell_name'].unique().tolist() # 228

# integrate
name_res = []
symbol_res = []
for cell in cell_names:
    tmp_df = target_ref[target_ref['cell_name']==cell]
    symbols = tmp_df['Symbol'].dropna().unique().tolist()
    if len(symbols) > 0:
        name_res.append(cell)
        symbol_res.append(symbols)
    else:
        pass
blood_dic = dict(zip(name_res,symbol_res)) # 220 cell types
pd.to_pickle(blood_dic,'/workspace/github/GLDADec/data/GSE65133/230524/human_pb_220_dic.pkl')

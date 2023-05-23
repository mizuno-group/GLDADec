# -*- coding: utf-8 -*-
"""
Created on 2023-05-18 (Thu) 09:57:08

marker prep for liver and blood comprehensive cell types

@author: I.Azuma
"""
#%%
import pandas as pd
import codecs
import collections
import itertools

Base_dir = '/workspace/github/GLDADec' # cloning repository

#%%
# load total data
with codecs.open(Base_dir + '/data/domain_info/CellMarker/Cell_marker_All.csv', "r", "Shift-JIS", "ignore") as file:
    total_ref = pd.read_table(file, delimiter=",")

# mouse
mouse_ref = total_ref[total_ref["species"].isin(["Mouse"])] # 35197
target_tissues = ['Blood','Liver'] # blood and liver
target_ref = mouse_ref[mouse_ref['tissue_class'].isin(target_tissues)]
target_ref = target_ref[target_ref['cell_type']=='Normal cell']

cell_names = target_ref['cell_name'].unique().tolist() # 137

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
liver_blood_dic = dict(zip(name_res,symbol_res)) # 116 cell types
pd.to_pickle(liver_blood_dic,'/workspace/github/GLDADec/data/mouse_liver_injury/230518/liver_blood_all_116_dic.pkl')

#%% records contain eosinophil
eos_ref = mouse_ref[mouse_ref['cell_name']=='Eosinophil']
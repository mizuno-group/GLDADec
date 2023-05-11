# -*- coding: utf-8 -*-
"""
Created on 2023-05-11 (Thu) 13:04:08

Marker preparation for all cell types

@author: I.Azuma
"""
#%%
import pandas as pd
import codecs
import collections

Base_dir = '/workspace/github/GLDADec' # cloning repository

#%%
# load total data
with codecs.open(Base_dir + '/data/domain_info/CellMarker/Cell_marker_All.csv', "r", "Shift-JIS", "ignore") as file:
    total_ref = pd.read_table(file, delimiter=",")

# mouse
mouse_ref = total_ref[total_ref["species"].isin(["Mouse"])] # 35197
tissues = mouse_ref['tissue_class'].unique().tolist()
tmp = mouse_ref[mouse_ref['tissue_type']=='Peripheral blood']

#%%
#target_tissues = ['Blood','Blood vessel','Liver','Spleen']
target_tissues = ['Liver']
target_ref = mouse_ref[mouse_ref['tissue_class'].isin(target_tissues)]
target_ref = target_ref[target_ref['cell_type']=='Normal cell']

cell_names = target_ref['cell_name'].unique().tolist()

symbol_res = []
for cell in cell_names:
    tmp_df = target_ref[target_ref['cell_name']==cell]
    symbols = tmp_df['Symbol'].dropna().unique().tolist()
    symbol_res.append(symbols)

liver_all_dic = dict(zip(cell_names,symbol_res))

#%%
pd.to_pickle(liver_all_dic,'/workspace/github/GLDADec/data/mouse_liver_injury/230511/liver_all_95_dic.pkl')
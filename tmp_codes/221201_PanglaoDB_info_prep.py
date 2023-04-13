# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 17:04:59 2022

@author: I.Azuma
"""
import pandas as pd
from matplotlib_venn import venn2
import matplotlib.pyplot as plt

#%%
total_ref = pd.read_table('D:/GdriveSymbol/datasource/Deconvolution/PanglaoDB/PanglaoDB_markers_27_Mar_2020.tsv')

cell_types = total_ref["cell type"].unique().tolist()

#%%
cell_names = ["Monocytes","NK cells","B cells naive","B cells memory","T cells","Gamma delta T cells"]
total_marker = []
for cell_name in cell_names:
    print("---",cell_name,"---")
    #tissue_type = ["Spleen","Blood","Bone marrow"]
    tissue_type = ["Blood"]
    species = ["Hs","Mm Hs"] # human
    target_ref = total_ref[total_ref["cell type"]==cell_name]
    target_ref = target_ref[target_ref["species"].isin(species)]
    marker = target_ref["official gene symbol"].tolist()
    unique_marker = list(set(marker))
    total_marker.append(unique_marker)

#%% summarize the result
panglao_dic = dict(zip(cell_names,total_marker))

import itertools
import collections

merge_marker = list(itertools.chain.from_iterable(list(panglao_dic.values())))
count_dic = dict(collections.Counter(merge_marker))
sort_count = sorted(count_dic.items(),key=lambda x : x[1])
unique_marker = []
for t in sort_count:
    if t[1] == 1:
        unique_marker.append(t[0])
    else:
        pass
    
# extract specific marker
b = []
for t in total_marker:
    b.append(list(set(t) & set(unique_marker)))
panglao_spe_dic = dict(zip(cell_names,b))

# save
pd.to_pickle(panglao_dic,'D:/GdriveSymbol/notebook/dry/Deconv_Method_dev/221201_method_dev_final_phase/221201_marker_prep/221201_human_marker/results/forGSE65133/human_blood_Panglao_6cell_raw_dic.pkl')
pd.to_pickle(panglao_spe_dic,'D:/GdriveSymbol/notebook/dry/Deconv_Method_dev/221201_method_dev_final_phase/221201_marker_prep/221201_human_marker/results/forGSE65133/human_blood_Panglao_6cell_specific_dic.pkl')

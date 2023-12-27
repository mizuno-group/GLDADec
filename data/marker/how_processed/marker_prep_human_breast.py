# -*- coding: utf-8 -*-
"""
Created on 2023-06-08 (Thu) 17:01:08

BRCA marker prep from CellMarker

@author: I.Azuma
"""
#%%
import codecs
import pandas as pd
import collections
import itertools

Base_dir = '/workspace/github/GLDADec'
#%%
# load total data
with codecs.open(Base_dir + '/data/marker/raw_info/CellMarker/Cell_marker_All.csv', "r", "Shift-JIS", "ignore") as file:
    total_ref = pd.read_table(file, delimiter=",")

# human
human_ref = total_ref[total_ref["species"].isin(["Human"])] # 60877

# %%
target_tissues = ['Breast']
target_ref = human_ref[human_ref['tissue_class'].isin(target_tissues)] # 1155

cell_names = target_ref['cell_name'].unique().tolist()
symbol_res = []
for cell in cell_names:
    tmp_df = target_ref[target_ref['cell_name']==cell]
    symbols = tmp_df['Symbol'].dropna().unique().tolist()
    symbol_res.append(symbols)

all_dic = dict(zip(cell_names,symbol_res))

#%% curation
remove_cells = ['Epithelial cell','Epithelial progenitor cell','Progenitor cell','Luminal progenitor cell','Basal progenitor cell','Endothelial progenitor cell','Immune cell','Endothelial cell','Cancer cell','Cycling epithelial cell','Proliferative cell','T cell','Lymphocyte','Myeloid cell','Fibroblast',]
merge_cells = [['Myoepithelial cell','Myo-epithelial cell'],['CD4+ T cell','Naive CD4 T cell'],['CD8+ T cell','Effector CD8 T cell','Exhausted CD8+ T cell'],['B cell','Naive memory B cell','Tumor-infiltrating B cell'],['Macrophage', 'Pan-macrophage','M1 macrophage','M2 macrophage','Tumor‐associated macrophage (TAM)'],['Dendritic cell','Myeloid dendritic cell','Plasmacytoid dendritic cell(pDC)'],['GrB+ Regulatory B cell', 'B10 Regulatory B cell'],['Neutrophil','Leukocyte'],['Mammary stem cell','Stem cell','EMT non-cancer stem cell(EMT non-CSC)'],['Cancer stem cell','EMT cancer stem cell','Breast cancer stem-like cell','Epithelial-mesenchymal transition cancer stem cell'],['Perivascular cell', 'Perivascular-like cell'],['Luminal cell','Mature luminal cell'],['T helper(Th) cell','T helper 1(Th1) cell','T follicular helper(Tfh) cell'],['Myofibroblast','Cancer-associated fibroblast','Antigen presentation cancer-associated fibroblast']]

'Myoepithelial cell', 'Basal epithelial cell', 'Luminal epithelial cell', 'Cancer stem cell', 'Mesenchymal stem cell', 'Hematopoietic stem cell', 'Natural killer cell', 'Cytotoxic T cell', 'Mesenchymal cell', 'B cell', 'Cancer-associated fibroblast', 'Regulatory T(Treg) cell', 'Macrophage', 'Basal cell', 'Luminal cell', 'Tumor-infiltrating T cell', 'Naive T(Th0) cell', 'Monocyte', 'Plasma cell',  'CD4+ T cell', 'CD8+ T cell', 'Red blood cell (erythrocyte)', 'Hematopoietic cell', 'Mammary stem cell', 'Conventional T(Tconv) cell', 'Mast cell', 'Gamma delta(γδ) T cell', 'Paget cell', 'Melanocyte', 'Suprabasal epithelial cell', 'Lymphatic endothelial cell', 'Proliferating T cell', 'Natural killer T(NKT) cell', 'GrB+ Regulatory B cell', 'Dendritic cell', 'Stromal cell', 'Polyploid giant cancer cell', 'Perivascular cell', 'Pericyte', 'Exhausted T(Tex) cell', 'Neutrophil', 'Tumor-initiating cell', 'Plasmablast', 'Adipocyte', 'Muscle cell', 'T helper(Th) cell', 'Eosinophil'

# process dict
# remove
use_k = []
use_v = []
for i,k in enumerate(all_dic):
    if k in remove_cells:
        pass
    else:
        use_k.append(k)
        use_v.append(all_dic.get(k))
removed_dic = dict(zip(use_k,use_v))

# merge
merge_targets = list(itertools.chain.from_iterable(merge_cells))
other_merge = []
for i,k in enumerate(removed_dic):
    if k in merge_targets:
        pass
    else:
        other_merge.append(k)
merged_name = []
merged_v = []
for m in merge_cells:
    merged_name.append(m[0])
    tmp = []
    for t in m:
        tmp.extend(removed_dic.get(t))
    merged_v.append(tmp)
merged_dic = dict(zip(merged_name,merged_v))

# create
final_k = []
final_v = []
for i,k in enumerate(all_dic):
    if k in other_merge:
        final_k.append(k)
        final_v.append(removed_dic.get(k))
    elif k in merged_name:
        final_k.append(k)
        final_v.append(merged_dic.get(k))
    else:
        print(k)
final_dic = dict(zip(final_k,final_v))

# save
# pd.to_pickle(final_dic,'/workspace/github/GLDADec/data/marker/human_breast_CellMarker.pkl')
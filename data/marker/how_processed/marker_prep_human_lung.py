# -*- coding: utf-8 -*-
"""
Created on 2023-06-12 (Mon) 16:54:37

Marker prep for LUAD

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

#%%
target_tissues = ['Lung']
target_ref = human_ref[human_ref['tissue_class'].isin(target_tissues)] # 1155

cell_names = target_ref['cell_name'].unique().tolist()
symbol_res = []
for cell in cell_names:
    tmp_df = target_ref[target_ref['cell_name']==cell]
    symbols = tmp_df['Symbol'].dropna().unique().tolist()
    symbol_res.append(symbols)

all_dic = dict(zip(cell_names,symbol_res))
# %% curation
remove_cells = ['Epithelial cell','Epithelial progenitor cell','Lung epithelial cell', 'Alveolar epithelial progenitor cell','Myeloid cell', 'T cell','OxPhos monocyte','Monocyte-derived cell','Mesothelial cell', 'Cancer cell','Endothelial cell','Secretory cell','T helper cell', 'SLC16A7+ cell','FOXN4+ cell','Mesenchymal progenitor cell','Alveolar cell','Immune cell','Dividing cell','Stem cell','Basal-like cell','Dysfunctional T cell','Lymphocyte','Proliferative cell','Neuron','Naive-like T cell', 'Activated T cell','Double-negative T cell','Lymphatic endothelial cell','Intrinsic neuron','Stromal cell','Basal epithelial cell','Progenitor cell','Pan-T cell','Systemic?venous endothelial cell','Anti-tumor immune cell','Eomesodermin homolog(EOMES)+ regulatory T cell type 1','CD45+ immune cell', 'Cycling cell','Basal cell']

merge_cells = [['Dendritic cell','Myeloid dendritic cell','Plasmacytoid dendritic cell','Plasmacytoid dendritic cell(pDC)','Conventional dendritic cell 1(cDC1)','Conventional dendritic cell(cDC)','Conventional dendritic cell 2(cDC2)','Activated dendritic cell', 'Myeloid dendritic cell 1','Migratory dendritic cell'],['Macrophage','M2 macrophage','M1 macrophage','Monocyte-derived macrophage','Epithelioid macrophage','Pro-inflammatory M1 macrophage','Mature macrophage','Interstitial macrophage','Pan-macrophage' ],['Tissue-resident macrophage','Alveolar macrophage'],['Mesothelioma cell', 'Malignant mesothelioma cell'],['Monocyte','Classical monocyte','Intermediate monocyte','Non-classical monocyte'],['Clara cell','Club cell (Clara cell)'],['Myofibroblast','Myofibroblastic cancer-associated fibroblast'],['CD8+ T cell','CD8 T cell','Memory CD8+ T cell', 'Effector CD8+ T cell','Pre-exhausted CD8+ T cell', 'Stem-like CD8+ T cell','Naive CD8+ T cell','Terminally differentiated CD8+ cell', 'Precursors CD8+ cell','Exhausted CD8+ T cell'],['Fibroblast','Cancer-associated fibroblast','Antigen presentation cancer-associated fibroblast','Lipofibroblast','Fibroblast-like cell','Inflammatory cancer-associated fibroblast'],['Cancer stem cell','Adenocarcinoma stem-like cell','Migrating cancer stem cell'],['Neuroendocrine cell','Pulmonary neuroendocrine cell'],['Ionocyte cell','Ionocyte'],['Mesenchymal cell','Mesenchymal stromal cell','Epithelial-mesenchymal cell', 'Mesenchymal stem cell'],['Neutrophil','White blood cell','Granulocyte','Leukocyte'],['Alveolar cell Type 1','Epithelial cell Type 1','Alveolar epithelial cell Type 1','Alveolar pneumocyte Type I'],['Alveolar cell Type 2','Epithelial cell Type 2','Alveolar type II (ATII) cell','Pulmonary alveolar cell Type 2','Alveolar epithelial cell Type 2','Alveolar type 2-like cell','Alveolar pneumocyte Type II','Type II pneumocyte'],['CD4+ T cell','CD4 T cell','Naive CD4+ T cell', 'Effector CD4+ T cell','Memory CD4+ T cell'],['Effector memory T cell','Terminal effector memory T cell'],['B cell','Naive B cell','Follicular B cell','Germinal center B cell','Memory B cell'],['Ciliated cell','Multiciliated cell', 'Ciliated airway epithelial cell'],['Lymphangioleiomyomatosis cell(LAMcore)','Noval lymphangioleiomyomatosis cell'],['Aerocyte','Aerocyte endothelial cell'],['Capillary cell','General capillary cell','Capillary endothelial cell']]

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
# pd.to_pickle(final_dic,'/workspace/github/GLDADec/data/marker/human_lung_CellMarker.pkl')
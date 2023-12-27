# -*- coding: utf-8 -*-
"""
Created on 2023-06-15 (Thu) 20:09:23

Marker prep for LIHC

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
target_tissues = ['Liver']
target_ref = human_ref[human_ref['tissue_class'].isin(target_tissues)] # 1155

cell_names = target_ref['cell_name'].unique().tolist()
symbol_res = []
for cell in cell_names:
    tmp_df = target_ref[target_ref['cell_name']==cell]
    symbols = tmp_df['Symbol'].dropna().unique().tolist()
    symbol_res.append(symbols)

all_dic = dict(zip(cell_names,symbol_res))
# %%
remove_cells = ['Fetal hepatocyte','Endothelial cell','T cell','Activated hepatic stellate cell','Ductal progenitor cell','Progenitor cell','Liver progenitor cell','Monocyte-dendritic cell progenitor cell', 'Macrophage-dendritic cell progenitor cell','Erythroid progenitor cell', 'Megakaryocyte-erythroid-mast cell progenitor cell', 'Myeloid progenitor cell','Neutrophil progenitor cell', 'Hematopoietic progenitor cell', 'Lymphoid progenitor cell', 'Hepatic progenitor cell','Endothelial progenitor cell','Stem cell','Cancer stem cell','Migrating cancer stem cell','Superpotent cancer stem cell', 'Putative hepatic stem cell','Hematopoietic precursor cell','Epithelial-mesenchymal cell','Liver bud hepatic cell','Scar‐associated macrophage (SAM)','Pan‐T cell','Exhausted T(Tex) cell','Biliary cell','Central memory T cell', 'Cancer cell','Lymphoid cell','Exhausted double-positive T cell', 'Cytotoxic double-positive T cell','Memory double-positive T cell', 'Naive double-positive T cell', 'Activated double-positive T cell','Double-positive Treg cell','Sinusoidal cell','Myeloid cell','Nerve cell','Granulocyte', 'T‐killer cell', 'Tissue resident cell','Innate lymphoid cell','Embryonic cell','Immune cell', 'Epithelial cell','Mesothelial cell','Pan leukocyte','Exhausted CD4+ T cell','Exhausted CD8+ T cell','Transitional exhausted CD8+ T cell','Regulatory T (Treg) cell','Regulatory T(Treg) cell','Circulating tumor cell','Mesenchymal cell',]

merge_cells = [['Hepatic stellate cell','Ito cell (hepatic stellate cell)'],['Macrophage', 'M1 macrophage','M2 macrophage'],['CD8+ T cell','CD8 T cell','Naive CD8+ T cell', 'Effector memory CD8+ T cell', 'Central memory CD8+ T cell'],['Dendritic cell','Conventional dendritic cell 2(cDC2)', 'Conventional dendritic cell 1(cDC1)','Mature dendritic cell','Plasmacytoid dendritic cell(pDC)'],['CD4+ T cell','Central memory CD4+ T cell','Naive CD4+ T cell'],['Natural killer T (NKT) cell','Natural killer T(NKT)-like cell','Natural killer T (NKT) cell',],['Fibroblast','Cancer-associated fibroblast'],['Erythroid cell','Red blood cell (erythrocyte)','Early erythrocyte','Primitive erythrocyte',],['Kupffer cell','Kupffer-like cell'],['Cytotoxic T cell','Cytotoxic CD4+ T cell','Cytotoxic T helper 1 cell'],['Neutrophil','Leukocyte'],['Hepatocyte','Hepatic cell'],['Myeloid-derived suppressor cell','Myeloid derived suppressor cell (MDSC)'],['Natural killer cell','Circulating natural killer cell']]


'Myofibroblast',  'B cell',   'EBV+ B lymphoma cell', 'Bile duct cell',  'Hepatoblast',  'Mesenchymal stem cell',  'Hematopoietic cell', 'Liver stem cell', 'Endothelial precursor cell',      'Monocyte',  'Memory B cell', 'Mesenchymal stromal cell', 'Platelet',    'Lymphoblast', 'Erythroblast',  'Mucosal-associated invariant T cell',  'T helper cell',   'Memory T cell', 'Cholangiocyte',      'Biliary epithelial cell', 'Naive T(Th0) cell', 'Mast cell',   'Tumor-initiating cell',    'Oval cell', 'Effector memory T cell',   'Naive B cell', 'T follicular helper(Tfh) cell', 'Plasma cell', 'Naive CD4 T cell',      'Sinusoidal endothelial cell',  'Megakaryocyte', 'Artery cell',   'Cardiovascular cell', 'Vascular endothelial cell', 'Hematopoietic stem cell',   'Portal vein cell',    'Septum transversumal cell(STC)',  'TIM-1+ Regulatory B cell', 'PD-1hi Regulatory B cell', 'Plasmablast', 'Memory CD27- B cell', 'Pro-B cell', 'T helper 17(Th17) cell', 'Pre-B cell', 'T helper 1(Th1) cell', 'Terminal differentiated B cell',  'Memory CD27+ B cell', 'Effector T(Teff) cell',       'Pericyte',  'Terminally differentiated effector memory T cell',   'Somatic stem cell', 'Gallbladder cell', 'Portal vein epithelial cell', 'lrNK cell', 'Portal fibroblast',  'Lipid-associated macrophage(LAM)',       'Suppressive regulatory T cell',     'T helper(Th) cell',   'Mucosa-associated invariant T (MAIT) cell',  'Resting regulatory T cell', 'Tumor‐associated macrophage (TAM)',  'Gamma delta(γδ) T cell',   'Beta cell(β cell)', 'Epsilon cell', 'Delta cell', 'Alpha cell (α cell)'

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
# pd.to_pickle(final_dic,'/workspace/github/GLDADec/data/marker/human_liver_CellMarker.pkl')

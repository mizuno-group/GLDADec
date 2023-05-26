# -*- coding: utf-8 -*-
"""
Created on 2023-05-24 (Wed) 17:58:45

Marker prep for mouse tissues other than liver

@author: I.Azuma
"""
#%%
import pandas as pd
import codecs
import collections
import itertools

Base_dir = '/workspace/github/GLDADec' # cloning repository

#%% Lung
# load total data
with codecs.open(Base_dir + '/data/domain_info/CellMarker/Cell_marker_All.csv', "r", "Shift-JIS", "ignore") as file:
    total_ref = pd.read_table(file, delimiter=",")

# mouse
mouse_ref = total_ref[total_ref["species"].isin(["Mouse"])] # 35197
target_tissues = ['Lung'] # lung
target_ref = mouse_ref[mouse_ref['tissue_class'].isin(target_tissues)]
target_ref = target_ref[target_ref['cell_type']=='Normal cell']

cell_names = target_ref['cell_name'].unique().tolist() # 138

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
lung_dic = dict(zip(name_res,symbol_res)) # 122 cell types

print(lung_dic.keys())
# remove cells
remove_cells = ['Epithelial cell', 'T cell','Macrophages-dendritic CD163+ cell','Neuroendocrine cell','Non-neuroendocrine cell', 'Myeloid cell', 'Endothelial cell', 'Immune cell','Mesenchymal progenitor cell','Lee et al.Cell.A', 'Lee et al.Cell.D', 'Lee et al.Cell.B', 'Lee et al.Cell.E', 'Lee et al.Cell.C','Endothelial progenitor cell', 'Progenitor cell', 'proliferative mesenchymal progenitor cell', 'Lymphocyte', 'Mesenchymal cell','Mesothelial cell','Group 2 innate lymphoid cell','Innate lymphoid cell', 'Group 3 innate lymphoid cell', 'Group 1 innate lymphoid cell','Mesenchymal stem cell','Stromal cell','Epithelial stem cell','Mature immune cell','Pan-endothelial cell', 'Proliferative endothelial cell','Damage-associated transient progenitor cell','Regulatory endothelial cell', 'Lymphoid cell','Mesenchymal stromal cell','Lymphocyte lineage','Systemic?venous endothelial cell','Vascular cell','Monocyte-derived cell','Distal lung progenitor cell', 'Distal epithelial cell',   'Pan-innate lymphoid cell','Proximal epithelial cell',]

# merge
merge_cells = [['Type I pneumocyte','Alveolar pneumocyte Type I','Early type I alveolar pneumocyte', 'Late type I alveolar pneumocyte','Mature type I alveolar pneumocyte',],['Type II pneumocyte','Alveolar pneumocyte Type II','Naive type II alveolar pneumocyte'],['Neutrophil','Polymorphonuclear neutrophil'],['Macrophage','Alveolar macrophage','M1 macrophage','M2 macrophage','Recruited airspace macrophage', 'Resident airspace macrophage','M1-like macrophage', 'Monocyte-derived macrophage','Interstitial macrophage','Lipid-associated macrophage(LAM)',],['Dendritic cell','Airway dendritic cell','Tolerogenic dendritic cell'],['Eosinophil','Activated eosinophil'],['Pericyte','Pericyte 2', 'Pericyte 1'],['Axin2+ cell', 'Axin2-Palpha+ cell'],[ 'Clara cell','Club cell (Clara cell)'],['Alveolar cell','Alveolar epithelial cell Type 1', 'Alveolar epithelial cell Type 2','Alveolar capillary cell','Differentiated alveolar epithelial cell','Alveolar type II (ATII) cell', 'Alveolar cell Type 1'],['Capillary cell','General capillary cell']]

# process dict
# remove
use_k = []
use_v = []
for i,k in enumerate(lung_dic):
    if k in remove_cells:
        pass
    else:
        use_k.append(k)
        use_v.append(lung_dic.get(k))
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
for i,k in enumerate(lung_dic):
    if k in other_merge:
        final_k.append(k)
        final_v.append(removed_dic.get(k))
    elif k in merged_name:
        final_k.append(k)
        final_v.append(merged_dic.get(k))
    else:
        print(k)
final_dic = dict(zip(final_k,final_v)) # 50
# save
pd.to_pickle(final_dic,'/workspace/github/GLDADec/data/mouse_liver_injury/230511/lung_merged_50_dic.pkl')

#%% Lung non-immune
lung_dic = pd.read_pickle('/workspace/github/GLDADec/data/mouse_liver_injury/230511/lung_merged_50_dic.pkl')
lung_non_immune = ['Clara cell','Type II pneumocyte','Type I pneumocyte','Myofibroblast', 'Lipofibroblast','Ciliated cell','Goblet cell', 'Matrix fibroblast', 'Fibroblast', 'Smooth muscle cell', 'Pericyte','Brush cell (Tuft cell)','Stem cell','Ionocyte', 'Mesoderm', 'Aerocyte','Glomerular capillaries cell', 'Vein cell', 'Artery cell','Megakaryocyte', 'Erythroblast', 'Venous cell', 'Capillary cell','Alveolar cell', 'Airway cell']
non_immune_markers = []
for cell in lung_non_immune:
    non_immune_markers.append(lung_dic.get(cell))
lung_non_immune_dic = dict(zip(lung_non_immune,non_immune_markers))
pd.to_pickle(lung_non_immune_dic,'/workspace/github/GLDADec/data/mouse_liver_injury/230511/lung_non_immune_25_from_50_dic.pkl')

"""
[  'Eosinophil', 'Macrophage',  'Dendritic cell',  'Neutrophil',   'Granulocyte', 'Axin2+ cell', 'Pdgfrapha+ cell', 'Wnt2+ cell', 'Lgr6+ cell', 'Lgr5+ cell', 'Natural killer T(NKT) cell',  'Regulatory T(Treg) cell',  'Tissue resident memory T(TRM) cell', 'CD8+ T cell', 'B cell', 'Gamma delta(γδ) T cell', 'T helper 2(Th2) cell', 'Monocyte', 'T helper 1(Th1) cell', 'Natural killer cell',   'Central memory T cell', 'Lymphatic endothelial cell', 'Basophil',  'γδ17 T cell', 'Inflammatory monocyte', ]
"""
#%% Kidney
# load total data
with codecs.open(Base_dir + '/data/domain_info/CellMarker/Cell_marker_All.csv', "r", "Shift-JIS", "ignore") as file:
    total_ref = pd.read_table(file, delimiter=",")

# mouse
mouse_ref = total_ref[total_ref["species"].isin(["Mouse"])] # 35197
target_tissues = ['Kidney'] # kidney
# FIXME: class contains other info (such as vessel)
#target_ref = mouse_ref[mouse_ref['tissue_class'].isin(target_tissues)] 
target_ref = mouse_ref[mouse_ref['tissue_type'].isin(target_tissues)]
target_ref = target_ref[target_ref['cell_type']=='Normal cell']

cell_names = target_ref['cell_name'].unique().tolist() # 116

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
kidney_dic = dict(zip(name_res,symbol_res)) # 102 cell types

print(kidney_dic.keys())
# remove cells
remove_cells = ['Stem cell','T cell','Neuronal Cell (Axon Only)','Endothelial cell','Endothelial cell (Lymphatic)','Transitional epithelial cell','Intercalated cell','Epithelial cell','Tubular cell','Principal cell', 'Renal progenitor cell',    'Mesenchymal progenitor cell','Stromal cell','Mesenchymal cell','Immune cell','Lymphocyte','Progenitor cell',]

# merge
merge_cells = [['Collecting duct principal cell','Collecting duct intercalated cell','Collecting duct transient cell','Collecting Duct Principal Cell','Inner Medullary Collecting Ductal cell'],['Medullary cell','Interstitial Cell (Medullary)'], ['Macrophage', 'M2 macrophage', 'M1 macrophage','R2b macrophage'],['Neutrophil','Leukocyte'],['Dendritic cell','Conventional dendritic cell 1(cDC1)', 'Plasmacytoid dendritic cell(pDC)', 'Monocyte-derived dendritic cell','Conventional dendritic cell(cDC)'],['General proximal cell','S1 proximal cell','S2 proximal cell','S3 proximal cell','S3 proximal tubule cell','S1/S2 proximal tubule cell','Proximal tubule cell','Proximal tubular cell','Proximal convoluted tubular cell', 'Proximal straight tubular cell'],['Distal tubular cell','Distal convoluted tubular cell'],['B cell','Follicular B cell'],['Cortical cell','Interstitial Cell (Cortical)'],['Polymorphonuclear Leukocyte','Polymorphonuclear leukocyte']]

# process dict
# remove
use_k = []
use_v = []
for i,k in enumerate(lung_dic):
    if k in remove_cells:
        pass
    else:
        use_k.append(k)
        use_v.append(lung_dic.get(k))
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
for i,k in enumerate(lung_dic):
    if k in other_merge:
        final_k.append(k)
        final_v.append(removed_dic.get(k))
    elif k in merged_name:
        final_k.append(k)
        final_v.append(merged_dic.get(k))
    else:
        print(k)
final_dic = dict(zip(final_k,final_v)) # 63
# save
pd.to_pickle(final_dic,'/workspace/github/GLDADec/data/mouse_liver_injury/230511/kidney_merged_63_dic.pkl')

#%% Kidney non-immune
kidney_dic = pd.read_pickle('/workspace/github/GLDADec/data/mouse_liver_injury/230511/kidney_merged_63_dic.pkl')
"""
[ 'Natural killer cell', 'B cell',  'Neutrophil', 'Macrophage',  'Dendritic cell', 'Mast cell', 'T helper(Th) cell',  'Regulatory T(Treg) cell', 'Eosinophil',  'Polymorphonuclear Leukocyte',  'Monocyte', 'Granular cell of afferent arteriole', 'Basophil',  'Natural killer T(NKT) cell', 'Plasma cell',  'Intercalated B cell', 'Memory B cell',  'Myeloid derived suppressor cell (MDSC)',  'Regulatory innate lymphoid cell',  'CD8+ T cell', 'Neural precursor cell', 'Effector memory T cell', 'Naive T(Th0) cell', 'Cytotoxic CD4+ T cell', 'Mononuclear phagocyte', ]
"""
kidney_non_immune = ['Oligodendrocyte', 'Medullary cell', 'Cortical cell', 'Mesangial cell', 'Collecting duct principal cell', 'Podocyte', 'Vascular endothelial cell','Fibroblast','Nephron progenitor cell', 'Connecting tubular cell', 'General proximal cell', 'Short Loop Descending Limb cell','Smooth muscle cell','Pericyte','Thin Ascending Limb cell','Megakaryocyte', 'Red blood cell (erythrocyte)','Long Descending Limb (Outer Medulla) cell', 'Myofibroblast','Macula densa cell', 'Intercalated A cell', 'Thick Ascending Limb cell', 'Glomerular cell','Loop of Henle cell','Distal tubular cell', 'Renal corpusle cell', 'Parietal epithelial cell(PEC)', 'Tubulointerstitial cell', 'Ascending loop of Henle cell', 'Actively dividing cell', 'Tubular epithelial cell(TEC)', 'Intermediate parietal epithelial cell (PEC)', 'Cultured parietal epithelial cell (PEC)', 'Extracellular matrix (ECM)-producing cell', 'Renin cell', 'Vascular smooth muscle cell(VSMC)','Thick ascending loop of Henle(TALH) cell', 'Intraglomerular mesangial cell']

non_immune_markers = []
for cell in kidney_non_immune:
    non_immune_markers.append(kidney_dic.get(cell))
kidney_non_immune_dic = dict(zip(kidney_non_immune,non_immune_markers))
pd.to_pickle(kidney_non_immune_dic,'/workspace/github/GLDADec/data/mouse_liver_injury/230511/kidney_non_immune_38_from_63_dic.pkl')
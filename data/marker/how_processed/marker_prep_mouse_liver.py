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
import itertools

Base_dir = '/workspace/github/GLDADec' # cloning repository

#%%
# load total data
with codecs.open(Base_dir + '/data/marker/how_processed/raw_info/CellMarker/Cell_marker_All.csv', "r", "Shift-JIS", "ignore") as file:
    total_ref = pd.read_table(file, delimiter=",")

# mouse
mouse_ref = total_ref[total_ref["species"].isin(["Mouse"])] # 35197
tissues = mouse_ref['tissue_class'].unique().tolist()
tmp = mouse_ref[mouse_ref['tissue_type']=='Peripheral blood']

#%%
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


for i,k in enumerate(liver_all_dic):
    print(k,liver_all_dic.get(k))
tmp = list(liver_all_dic.keys())

# remove cells
remove_cells = ['Adenocarcinoma cell','Alveolar cell', 'Ciliated cell', 'Cardiovascular cell', 'Cytotoxic T cell','Endothelial cell','Epithelial cell','Erythroid progenitor cell','Hematopoietic progenitor cell','Hematopoietic stem cell','Hepatic endothelial cell','Hepatic progenitor cell','Hepatobiliary cell','Immune cell','Liver progenitor cell','Lymphocyte','Lymphoid cell','Lymphoid progenitor cell','Macrophage-dendritic cell progenitor cell','Megakaryocyte-erythroid-mast cell progenitor cell','Mesothelial cell','Monocyte-dendritic cell progenitor cell','Myeloid cell','Myeloid progenitor cell','Neutrophil progenitor cell','Non-neuroendocrine cell','Pan lymphocyte','Pancreatobiliary cell','Primitive erythrocyte','Progenitor cell','Proximal tubular cell','Septum transversumal cell(STC)','Sinusoidal cell','Sinusoidal endothelial cell','Stromal cell','T cell','Mesenchymal cell']

# merge
merge_cells = [['Bile duct cell','Biliary cell','Biliary epithelial cell','Cholangiocyte','Ductal cell'],['Macrophage','Capsular macrophage','M1 macrophage','M2 macrophage','NASH-associated macrophage','Peritoneal macrophage'],['Dendritic cell','Conventional dendritic cell 1(cDC1)','Conventional dendritic cell 2(cDC2)','Migratory (or mature) dendritic cell(MDC)'],['Effector memory T cell','Effector T(Teff) cell'],['Hepatic stellate cell','Stellate cell','Ito cell (hepatic stellate cell)'],['Hepatocyte','Hepatocellular cell','Mature hepatocyte','Periportal hepatocyte'],['Neutrophil','Leukocyte'],['Liver sinusoid endothelial cell(LSECs)','Liver sinusoidal endothelial cell'],['Fibroblast','Myofibroblast','Portal fibroblast'],['Stem cell','Liver stem cell']]
# process dict
# remove
use_k = []
use_v = []
for i,k in enumerate(liver_all_dic):
    if k in remove_cells:
        pass
    else:
        use_k.append(k)
        use_v.append(liver_all_dic.get(k))
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
for i,k in enumerate(liver_all_dic):
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
# pd.to_pickle(final_dic,Base_dir+'/data/marker/mouse_liver_CellMarker.pkl')

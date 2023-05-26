# -*- coding: utf-8 -*-
"""
Created on 2023-05-25 (Thu) 23:04:57

Extract robust immune markers from multiple tissues.

@author: I.Azuma
"""
#%%
import pandas as pd
import codecs
import collections
import itertools
import collections

Base_dir = '/workspace/github/GLDADec' # cloning repository

#%% liver
# load total data
with codecs.open(Base_dir + '/data/domain_info/CellMarker/Cell_marker_All.csv', "r", "Shift-JIS", "ignore") as file:
    total_ref = pd.read_table(file, delimiter=",")

# mouse
mouse_ref = total_ref[total_ref["species"].isin(["Mouse"])] # 35197
print(sorted(mouse_ref["tissue_class"].unique().tolist()))
"""
'Abdomen', 'Adipose tissue', 'Airway', 'Anorectal junction', 'Aorta', 'Arm', 'Artery', 'Articulation', 'Belly', 'Bladder', 'Blood', 'Blood vessel', 'Bone', 'Bone marrow', 'Brain', 'Breast', 'Cartilage', 'Cerebellum', 'Cerebral organoid', 'Colon', 'Connective tissue', 'Diaphragm', 'Dorsal root ganglia', 'Ear', 'Embryo', 'Embryos', 'Endocardium', 'Endometrium', 'Epithelium', 'Esophagus', 'Eye', 'Fetal brain', 'Fetal heart', 'Flesh', 'Foot', 'Gastrointestinal tract', 'Gingiva', 'Gonad', 'Gut', 'Head and Neck', 'Heart', 'Hind limb', 'Hindlimb', 'Intestine', 'Intestine/Proliferating ECs pool', 'Kidney', 'Knee', 'limb', 'Limb', 'Liver', 'Lumbar', 'Lung', 'Lymph', 'lymph node', 'Lymph node', 'Lymphoid tissue', 'Macrovessel', 'Mammary gland', 'Mammary Gland', 'Meniscus', 'Mouth', 'Muscle', 'Nerve', 'Neural tube', 'Nodose', 'Non-Vasculature', 'Nose', 'Omentum', 'Oral cavity', 'Ovary', 'Pancreas', 'PeriBiliary cell gland', 'Peribiliary gland', 'Peritoneal cavity', 'Peritoneum', 'Peyer patch', 'Pharynx', 'Placenta', 'Prostate', 'Pylorus', 'Renal Tubule', 'Salivary gland', 'Scalp', 'Skeletal muscle', 'Skeletal Muscle', 'Skin', 'Soft tissue', 'Spinal column', 'Spinal cord', 'Spleen', 'Stomach', 'Suprarenal gland', 'Sural nerve', 'Tendon', 'Testis', 'Thymus', 'Thyroid', 'Tongue', 'Tonsil', 'Tooth', 'Trachea', 'Umbilical cord', 'Undefined', 'Uterine cervix', 'Uterus', 'Vagina', 'Vein'
"""

#%%
# liver
liver_ref = mouse_ref[mouse_ref["tissue_class"].isin(["Liver"])]
liver_ref = liver_ref[liver_ref['cell_type']=='Normal cell']
liver_cells = liver_ref["cell_name"].unique().tolist()
# lung
lung_ref = mouse_ref[mouse_ref["tissue_class"].isin(["Lung"])]
lung_ref = lung_ref[lung_ref['cell_type']=='Normal cell']
lung_cells = lung_ref["cell_name"].unique().tolist()
# kidney
kidney_ref = mouse_ref[mouse_ref["tissue_class"].isin(["Kidney"])]
kidney_ref = kidney_ref[kidney_ref['cell_type']=='Normal cell']
kidney_cells = kidney_ref["cell_name"].unique().tolist()
# brain
brain_ref = mouse_ref[mouse_ref["tissue_class"].isin(["Brain"])]
brain_ref = brain_ref[brain_ref['cell_type']=='Normal cell']
brain_cells = brain_ref["cell_name"].unique().tolist()
# spleen
spleen_ref = mouse_ref[mouse_ref["tissue_class"].isin(["Spleen"])]
spleen_ref = spleen_ref[spleen_ref['cell_type']=='Normal cell']
spleen_cells = spleen_ref["cell_name"].unique().tolist()

# integrate
total = list(itertools.chain.from_iterable([liver_cells,lung_cells,kidney_cells,brain_cells,spleen_cells]))
count_dic = dict(collections.Counter(total))
print(count_dic)

#%%
import matplotlib.pyplot as plt
multi_cell = []
multi_count = []
for i,k in enumerate(count_dic):
    n = count_dic.get(k)
    if n>=3:
        multi_cell.append(k)
        multi_count.append(n)
    else:
        pass

multi_dic = dict(zip(multi_cell,multi_count))
for i,k in enumerate(multi_dic):
    print(k," ",multi_dic.get(k))
"""
T cell   5
CD4+ T cell   4
Dendritic cell   5
Progenitor cell   5
Stem cell   4
Natural killer cell   5
B cell   5
Monocyte   5
Endothelial cell   5
Epithelial cell   4
Mesenchymal cell   4
Neutrophil   5
Macrophage   5
Myeloid cell   5
Basophil   4
M1 macrophage   4
M2 macrophage   4
Fibroblast   4
Lymphocyte   5
Natural killer T(NKT) cell   4
CD8+ T cell   4

Myofibroblast   3
Monocyte-derived macrophage   3
Leukocyte   3
Vascular endothelial cell   3
Megakaryocyte   3
Artery cell   3
Microglial cell   3
Immune cell   3
Stromal cell   3
Conventional dendritic cell 1(cDC1)   3
Conventional dendritic cell 2(cDC2)   3
Naive T(Th0) cell   3
Effector memory T cell   3
Vascular smooth muscle cell(VSMC)   3
Eosinophil   3
Smooth muscle cell   3
Pericyte   3
Granulocyte   3
Regulatory T(Treg) cell   3
Astrocyte   3
T helper(Th) cell   3
Plasma cell   3
Plasmacytoid dendritic cell(pDC)   3
Monocyte-derived dendritic cell   3
"""
freq_immunes = ["CD4+ T cell","Dendritic cell","Natural killer cell","B cell","Monocyte","Neutrophil","Macrophage","Basophil","Natural killer T(NKT) cell","CD8+ T cell","Naive T(Th0) cell","Effector memory T cell","Eosinophil","Regulatory T(Treg) cell","T helper(Th) cell"]

#%%
cell = "Basophil"
liver_marker = liver_ref[liver_ref['cell_name']==cell]['Symbol'].dropna().unique().tolist()
lung_marker = lung_ref[lung_ref['cell_name']==cell]['Symbol'].dropna().unique().tolist()
kidney_marker = kidney_ref[kidney_ref['cell_name']==cell]['Symbol'].dropna().unique().tolist()
brain_marker = brain_ref[brain_ref['cell_name']==cell]['Symbol'].dropna().unique().tolist()
spleen_marker = spleen_ref[spleen_ref['cell_name']==cell]['Symbol'].dropna().unique().tolist()

total_marker = list(itertools.chain.from_iterable([liver_marker,lung_marker,kidney_marker,brain_marker,spleen_marker]))
marker_count_dic = dict(collections.Counter(total_marker))

multi_marker = []
multi_count = []
for i,k in enumerate(marker_count_dic):
    n = marker_count_dic.get(k)
    if n>=2:
        multi_marker.append(k)
        multi_count.append(n)
    else:
        pass

#%%
def robust_marker_collection(cell='T cell'):
    liver_marker = liver_ref[liver_ref['cell_name']==cell]['Symbol'].dropna().unique().tolist()
    lung_marker = lung_ref[lung_ref['cell_name']==cell]['Symbol'].dropna().unique().tolist()
    kidney_marker = kidney_ref[kidney_ref['cell_name']==cell]['Symbol'].dropna().unique().tolist()
    brain_marker = brain_ref[brain_ref['cell_name']==cell]['Symbol'].dropna().unique().tolist()
    spleen_marker = spleen_ref[spleen_ref['cell_name']==cell]['Symbol'].dropna().unique().tolist()

    total_marker = list(itertools.chain.from_iterable([liver_marker,lung_marker,kidney_marker,brain_marker,spleen_marker]))
    marker_count_dic = dict(collections.Counter(total_marker))

    multi_marker = []
    multi_count = []
    for i,k in enumerate(marker_count_dic):
        n = marker_count_dic.get(k)
        if n>=2:
            multi_marker.append(k)
            multi_count.append(n)
        else:
            pass
    return multi_marker

robust_markers = []
for cell in freq_immunes:
    res = robust_marker_collection(cell=cell)
    robust_markers.append(res)
    print(cell,' ',len(res))

robust_dic = dict(zip(freq_immunes,robust_markers))
"""
CD4+ T cell   1
Dendritic cell   6
Natural killer cell   24
B cell   14
Monocyte   7
Neutrophil   19
Macrophage   40
Basophil   0
Natural killer T(NKT) cell   1
CD8+ T cell   3
Naive T(Th0) cell   3
Effector memory T cell   0
Eosinophil   0
Regulatory T(Treg) cell   3
T helper(Th) cell   0
"""
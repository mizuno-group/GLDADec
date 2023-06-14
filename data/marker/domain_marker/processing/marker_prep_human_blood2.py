# -*- coding: utf-8 -*-
"""
Created on 2023-06-14 (Wed) 17:40:17

230614 marker preparation for human blood cells

@author: I.Azuma
"""
#%%
import pandas as pd
import codecs
import collections
import seaborn as sns
import matplotlib.pyplot as plt
import collections

#%%
# load total data
with codecs.open("/workspace/github/GLDADec/data/marker/raw_info/CellMarker/Cell_marker_All.csv", "r", "Shift-JIS", "ignore") as file:
    total_ref = pd.read_table(file, delimiter=",")

#%% PBMC
human_ref = total_ref[total_ref["species"].isin(["Human"])] # 60877
pbmc_ref = human_ref[human_ref["tissue_type"].isin(["Peripheral blood"])] # 2731

# Monocyte
mon_ref = pbmc_ref[pbmc_ref["cell_name"].isin(["Monocyte"])]
sym_mon = mon_ref["Symbol"].dropna().unique().tolist()
mon_facs= ['CD14','CD16','CSF1R','CX3CR1','ITGAM','ITGAX','HLA-DR','CCR2','XCXR4','FCGR1A','CD86',
             'PTPRC','IL3RA','CD27','CCR5','CD32','CD1A','MRC1','ITGB3','CD9','CXCR6','CCR1','FLT3',
             'CLEC12A','CCR6','CD68','KIT','CD1C','TEK'] # (biocompare)
mon_marker = sorted(list(set(sym_mon) & set(mon_facs)))

# Neutrophil
neu_ref = pbmc_ref[pbmc_ref["cell_name"].isin(["Neutrophil"])]
sym_neu = neu_ref["Symbol"].dropna().unique().tolist()
neu_facs= ['CCR7','CD14','CD177','CD24','CD47','CD63','CD86','CXCR1','CXCR2','CXCR4','FCGR3A','FLT1','ICAM1','IL17RA',
           'ITGA4','ITGAM','ITGAX','ITGB2','PECAM1','PTPRC','SELL','SPN','TLR2','TLR4','TLR5','TLR7','TLR8','TLR9'] # (biocompare)
neu_marker = sorted(list(set(sym_neu) & set(neu_facs)))

# NK
nk_ref = pbmc_ref[pbmc_ref["cell_name"].isin(["Natural killer cell"])]
sym_nk = nk_ref["Symbol"].dropna().unique().tolist()
nk_facs= ['CD56','CCR7','CSF2','CXCR3','IFNG','IL2RB','IL7R','KIT','KLRC1','KLRD1','NCR1','SELL',
            'CD16','CX3CR1','CXCR1','ITGB2','KIR','KLRC2','KLRG1','PRF1'] # (biocompare)
nk_marker = sorted(list(set(sym_nk) & set(nk_facs)))

# B
b_n_ref = pbmc_ref[pbmc_ref["cell_name"].isin(["Naive B cell","Resting naive B cell"])]
b_m_ref = pbmc_ref[pbmc_ref["cell_name"].isin(["Memory B cell","Resting memory B cell"])]
sym_bn = b_n_ref["Symbol"].dropna().unique().tolist()
sym_bm = b_m_ref["Symbol"].dropna().unique().tolist()
bnaive_marker = ['BCL7A','FCER2', 'IGHD', 'IGHM', 'PAX5', 'TCL1A']

# CD4
cd4_m_ref = pbmc_ref[pbmc_ref["cell_name"].isin(["Memory CD4+ T cell"])]
cd4_n_ref = pbmc_ref[pbmc_ref["cell_name"].isin(["Naive CD4 T cell","Naive CD4+ T cell"])]
sym_cd4n = cd4_n_ref["Symbol"].dropna().unique().tolist()
sym_cd4m = cd4_m_ref["Symbol"].dropna().unique().tolist()

# CD8
cd8_ref = pbmc_ref[pbmc_ref["cell_name"].isin(["Activated CD8+ T cell","Activated naive CD8+ T cell","Activated memory CD8+ T cell","CD8 T cell","CD8+ T cell","Memory CD8 T cell","Memory CD8+ T cell","Naive CD8 T cell","Naive CD8+ T cell"])]
sym_cd8 = cd8_ref["Symbol"].dropna().unique().tolist()
cd8_marker = sorted(['CD8A','CD8B'])

# Gamma delta
gd_ref = pbmc_ref[pbmc_ref["cell_name"].isin(["Gamma delta(γδ) T cell"])]
sym_gd = gd_ref["Symbol"].dropna().unique().tolist()
gd_marker = ['S100B','TRGV9','TRGV1','CCL5','STMN1','TRGJ2','HMGB2','TRGJP2','TRGV3','TRGV11','H2AFZ','NUSAP1','TROAP','TUBB']

# Dendritic cell
dc_ref = pbmc_ref[pbmc_ref["cell_name"].isin(["Dendritic cell"])]
sym_dc = dc_ref["Symbol"].dropna().unique().tolist()

# Treg
treg_ref = pbmc_ref[pbmc_ref["cell_name"].isin(['Regulatory T (Treg) cell','Regulatory T(Treg) cell'])]
sym_treg = treg_ref["Symbol"].dropna().unique().tolist()

#%% aggregate
a = [sym_mon,sym_neu,sym_nk,sym_bn,sym_bm,sym_cd4n,sym_cd4m,sym_cd8,sym_gd,sym_dc,sym_treg]
k = ["Monocytes","Neutrophils","NK cells","B cells naive","B cells memory","T cells CD4 naive","T cells CD4 memory","T cells CD8","T cells gamma delta","Dendritic cells","Treg"]

cellmarker_dic = dict(zip(k,a))
pd.to_pickle(cellmarker_dic,'/workspace/github/GLDADec/data/marker/domain_marker/results/v2/CellMarker_dic.pkl')

#%% reflect domain knowledge
b = [mon_marker,neu_marker,nk_marker,bnaive_marker,sym_bm,sym_cd4n,sym_cd4m,cd8_marker,gd_marker,sym_dc,sym_treg]
k = ["Monocytes","Neutrophils","NK cells","B cells naive","B cells memory","T cells CD4 naive","T cells CD4 memory","T cells CD8","T cells gamma delta","Dendritic cells","Treg"]

cellmarker_dic = dict(zip(k,b))
pd.to_pickle(cellmarker_dic,'/workspace/github/GLDADec/data/marker/domain_marker/results/v2/CellMarker_domain_dic.pkl')

# %%
'Effector T cell', 'Pan-T cell', 'T cell', 'CD14+ monocyte', 'Fibroblast', 'Ovalbumin-specific regulatory T cell', 'Activated naive CD8+ T cell', 'Cancer stem cell', 'Intermediate monocyte', 'CD4+ central memory like T (Tcm-like) cell', 'Dendritic cell', 'CD4+ tumor antigen-specific T (Tas) cell', 'Precursor cell', 'Stem memory T?cell', 'Lymphoid stem cell', 'Natural regulatory T (Treg) cell', 'Memory CD8 T cell', 'Myeloid-derived suppressor cell', 'Pan-B cell', 'Activated natural killer cell', 'Unswitched memory B cell(UnSw MB)', 'Regulatory B cell', 'Patelet', 'Conventional dendritic cell 2(cDC2)', 'CD4+ T helper cell', 'CD4-CD28- T cell', 'Resting naive B cell', 'Tissue resident memory T(TRM) cell', 'Conventional T(Tconv) cell', 'Cytotoxic NK cell', 'Activated naive T cell', 'Tumor regulatory T cell', 'Immature myeloid cell', 'Double-positive T cell', 'Activated CD8+ T cell', 'Circulating precursor cell', 'Cardiac-lineage stem cell', 'Immature erythroid cell', 'Antibody Secreting B cell', 'Exhausted T(Tex) cell', 'T helper cell', 'Neutrophil progenitor cell', 'Class-switched memory B cell', 'Polymorphonuclear neutrophil', 'Naive B cell', 'Mature dendritic cell', 'Circulating activated B cell', 'Induced regulatory T (Treg) cell', 'Lymphoid cell', 'Naive CD8 T cell', 'CD8+ tumor antigen-specific T (Tas) cell', 'Natural killer cell', 'Non-switched memory B cell', 'CD4+ cytotoxic T1 cell', 'Natural killer T (NKT) cell', 'Cancer cell', 'Intermediate monocyte cell', 'B cell lineage', 'Myeloid stem cell', 'CD4+CD25+ regulatory T cell', 'TolDC-induced regulatory T cell', 'T cell large granular lymphocytic leukemia cell', 'Aged memory B cell', 'Proliferative cell', 'Cytotoxic CD4+ T2 cell', 'CD56dim Natural killer cell', 'CD16 monocyte', 'Memory B cell', 'Proliferative lymphocyte', 'Memory CD4+ T cell', 'Conventional dendritic cell 1(cDC1)', 'Circulating stem like cell', 'Germinal center B cell', 'CD8+ T cell', 'Microglial cell', 'Non-classical monocyte', 'Proliferative T cell', 'CD25 LAG3+ T cell', 'Proliferative CD8+ T cell', 'Monocyte precursor', 'Activated memory T cell', 'Mast cell', 'Basophil', 'CD20+ B cell', 'Monocyte', 'Dividing plasma B cell', 'Innate lymphoid cell', 'CD4+ cytotoxic T2 cell', 'Naive T(Th0) cell', 'CD8 T cell', 'Neutrophil', 'Double-negative B cell', 'Abnormal plasma cell', 'Effector CD8+ T cell', 'Plasma cell', 'Leukocyte', 'Effector memory CD4 T cell', 'Immunoregulatory natural killer cell', 'B cell', 'Myeloid dendritic cell', 'T helper2 (Th2) cell', 'T helper1 (Th1) cell', 'Multipotent Stem(CiMS) cell', 'Memory CD8+ T cell', 'Red blood cell (erythrocyte)', 'Cardiac stem cell', 'Exhausted CD4+ T cell', 'Abnormal myeloid cell', 'Plasmablast', 'Memory-like B cell', 'Dermal cell', 'Marginal zone(MZ)-like B cell', 'Resting memory B cell', 'Effector CD8+ memory T (Tem) cell', 'Mesoderm cell', 'Mature adipocyte', 'Antibody-secreting cell', 'Polymorphonuclear myeloid-derived suppressor cell', 'Gastric progenitor cell', 'Large granular lymphocyte', 'Cytotoxic CD4+ T cell', 'Naive CD8+ T cell', 'Effector memory CD4+ T cell', 'Circulating memory T cell', 'Atypical B cell(ABC)', 'Mature B cell', 'Class switched plasmablast', 'Pro-Natural killer cell (pro-NK cell)', 'T helper17 (Th17) cell', 'Naive T cell', 'CD16+ dendritic cell', 'Classical monocyte', 'Exhausted CD8+ T cell', 'Endothelial precursor cell', 'Regulatory B(Breg) cell', 'Hematopoietic progenitor cell', 'Age-associated B cell', 'Macrophage', 'Activated memory CD8+ T cell', 'Effector T(Teff) cell', 'Activated B cell', 'CD14 monocyte', 'Endothelial progenitor cell', 'Polymorphonuclear myeloid-derived suppressor(PMN-MDSC) cell', 'Monocyte lineage', 'Plasmacytoid dendritic cell', 'Megakaryocyte progenitor cell', 'T helper 17(Th17) cell', 'Stem cell', 'Cycling cell', 'Peripheral immune cell', 'CD4 T cell', 'CD56bright Natural killer cell', 'Myeloid derived suppressor cell (MDSC)', 'Transitional B cell', 'Regulatory CD4 T cell', 'Suppressive monocyte', 'IgA memory B cell', 'T helper 2(Th2) cell', 'Regulatory T (Treg) cell', 'Platelet', 'Double-negative T cell', 'Memory T cell', 'Differentiated effector T cell', 'M1 macrophage', 'Circulating angiogenic cell', 'Esophageal progenitor cell', 'Naive CD4 T cell', 'Tissue resident macrophage', 'Lymphocyte', 'Revertant memory CD8+ T(TEMRA) cell', 'Epicardial progenitor cell', 'CD25+LAG3+ T cell', 'Tissue-like memory B cell', 'Mesenchymal stem cell', 'CD4+ T cell', 'CD14+CD16+ monocyte', 'Naive CD4+ T cell', 'Switched memory B cell(Sw MB)', 'Central memory T cell', 'Endothelial cell', 'Monocyte derived dendritic cell', 'Fully activated dendritic cell', 'IgG memory B cell', 'IgM only B cell', 'Circulating fetal cell', 'Erythroid cell', 'Hematopoietic stem cell', 'Monocytic myeloid-derived suppressor cell', 'Cytotoxic T cell', 'T helper(Th) cell', 'Activated T cell', 'Activated peripheral helper T cell', 'Activated CD25+ regulatory T cell', 'Mature progenitor cell', 'Gamma delta(γδ) T cell', 'T helper 1(Th1) cell', 'Double-negative memory B cell', 'M2 macrophage', 'Pro-inflammatory macrophage', 'Myeloid progenitor cell', 'Effector CD4+ T cell', 'Central memory CD4+ T cell', 'Activated CD4+ T cell', 'Mature neutrophil', 'IgM memory B cell', 'Regulatory CD4+ T cell', 'Mesenchymal cell', 'Lung progenitor cell', 'Responder T cell', 'Myeloid cell', 'Class switched memory B IgGκ cell', 'Intermediate transition memory B cell', 'CD4-CD28+ T cell', 'Granulocyte', 'Regulatory CD25+ T cell', 'Dendritic cell lineage', 'Effector memory CD8 T cell', 'Regulatory T(Treg) cell', 'CD16+ monocyte', 'Peripheral blood mononuclear cell (PBMC)', 'Suppressor T cell', 'T helper 9(Th9) cell', 'Myeloid dendritic cell 1', 'Circulating progenitor cell', 'Effector memory CD8+ T cell', 'Epithelial cell', 'Progenitor cell', 'CD8+?intraepithelial cell', 'IL-17Ralpha T cell', 'Natural killer T(NKT) cell', 'T follicular helper(Tfh) cell', 'Cytotoxic CD8 T cell', 'Regulatory CD8+ T cell', 'Unswitched plasmablast', 'Follicular cytotoxic CD8+ T cell', 'Plasmacytoid dendritic cell(pDC)', 'Proliferative CD4+ T cell', 'Effector memory T cell', 'Megakaryocyte', 'Central memory CD8+ T cell', 'Responding conventional T cell', 'Effector regulatory T(eTreg) cell', 'Eosinophil', 'Immature B cell', 'Mucosa-associated invariant T (MAIT) cell', 'Cytotoxic CD8+ T cell', 'Early effector T cell', 'Switched memory B cell', 'Naive regulatory T (Treg) cell', 'Atypical memory B cell', 'Circulating plasma cell'
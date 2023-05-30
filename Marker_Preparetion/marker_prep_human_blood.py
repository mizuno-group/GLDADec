# -*- coding: utf-8 -*-
"""
Created on 2023-05-29 (Mon) 16:52:56

marker preparation for human blood cells

@author: I.Azuma
"""
#%%
import pandas as pd

#%% GSE65133
# load dictionaries
cellmarker_dic = pd.read_pickle('/workspace/github/GLDADec/data/domain_info/human_PBMC_CellMarker_8cell_spe_dic_v1.pkl')
panglao_dic = pd.read_pickle('/workspace/github/GLDADec/data/domain_info/human_blood_Panglao_6cell_raw_dic.pkl')
lm22_df = pd.read_csv('/workspace/github/GLDADec/data/GSE65133/lm22_signature.csv',index_col=0)
lm22_deg_dic = pd.read_pickle('/workspace/github/GLDADec/data/GSE65133/LM22_max_deg_dic.pkl')

# Monocyte 
mon_candi = sorted(cellmarker_dic.get('Monocytes'))
mon_facs= ['CD14','CD16','CSF1R','CX3CR1','ITGAM','ITGAX','HLA-DR','CCR2','XCXR4','FCGR1A','CD86',
             'PTPRC','IL3RA','CD27','CCR5','CD32','CD1A','MRC1','ITGB3','CD9','CXCR6','CCR1','FLT3',
             'CLEC12A','CCR6','CD68','KIT','CD1C','TEK'] # (biocompare)
mon_marker = sorted(list(set(mon_candi) & set(mon_facs)))

# NK
nk_candi = sorted(cellmarker_dic.get('NK cells'))
nk_facs= ['CD56','CCR7','CSF2','CXCR3','IFNG','IL2RB','IL7R','KIT','KLRC1','KLRD1','NCR1','SELL',
            'CD16','CX3CR1','CXCR1','ITGB2','KIR','KLRC2','KLRG1','PRF1'] # (biocompare)
nk_marker = sorted(list(set(nk_candi) & set(nk_facs)))

# B naive
#bnaive_candi = sorted(cellmarker_dic.get('B cells naive')) # ['BCL7A', 'CD24', 'FCER2', 'IGHD', 'IGHM', 'IL4R', 'PAX5', 'TCL1A']
# FIXME: remove CD24 and IL4R
bnaive_marker = ['BCL7A','FCER2', 'IGHD', 'IGHM', 'PAX5', 'TCL1A']

# B memory
bmemory_marker = sorted(cellmarker_dic.get('B cells memory'))

# CD8
cd8_marker = sorted(['CD8A','CD8B'])

# CD4 naive
cd4naive_marker = sorted(cellmarker_dic.get('T cells CD4 naive'))

# CD4 memory
cd4memory_marker = sorted(cellmarker_dic.get('T cells CD4 memory'))

# Gamma delta
#gd_candi = sorted(panglao_dic.get('Gamma delta T cells'))
gd_marker = ['S100B','TRGV9','TRGV1','CCL5','STMN1','TRGJ2','HMGB2','TRGJP2','TRGV3','TRGV11','H2AFZ','NUSAP1','TROAP','TUBB']

# integrate and generate marker dict
k = ['Monocytes', 'NK cells', 'B cells naive', 'B cells memory', 'T cells CD4 naive', 'T cells CD4 memory', 'T cells CD8', 'T cells gamma delta']
v = [mon_marker,nk_marker,bnaive_marker,bmemory_marker,cd4naive_marker,cd4memory_marker,cd8_marker,gd_marker]
domain_dic = dict(zip(k,v))
pd.to_pickle(domain_dic,'/workspace/github/GLDADec/data/GSE65133/domain/gse65133_domain_dic.pkl')

#%% GSE107572
pre_domain_dic = pd.read_pickle('/workspace/github/GLDADec/data/GSE65133/domain/gse65133_domain_dic.pkl') # defined above section
cellmarker_dic = pd.read_pickle('/workspace/github/GLDADec/data/GSE107572/CellMarker/human_PBMC_CellMarker_8cell_raw_dic_230307.pkl')

# B cells
bcandi = []
bcandi.extend(pre_domain_dic.get('B cells memory'))
bcandi.extend(pre_domain_dic.get('B cells naive'))

# CD4 cells
cd4candi = []
cd4candi.extend(pre_domain_dic.get('T cells CD4 memory'))
cd4candi.extend(pre_domain_dic.get('T cells CD4 naive'))

# CD8 cels
cd8candi = pre_domain_dic.get('T cells CD8')

# NK cells
nkcandi = pre_domain_dic.get('NK cells')

# Monocytes
moncandi = pre_domain_dic.get('Monocytes')

# Neutrophils
neucandi = cellmarker_dic.get('Neutrophils')
neu_facs= ['CCR7','CD14','CD177','CD24','CD47','CD63','CD86','CXCR1','CXCR2','CXCR4','FCGR3A','FLT1','ICAM1','IL17RA',
           'ITGA4','ITGAM','ITGAX','ITGB2','PECAM1','PTPRC','SELL','SPN','TLR2','TLR4','TLR5','TLR7','TLR8','TLR9'] # (biocompare)
neucandi = sorted(list(set(neucandi) & set(neu_facs)))

# Dendritic cells
dccandi = cellmarker_dic.get('Dendritic cells')
#dc_facs = ['CLEC4C','LILRB4','NRP1','CLEC10A','CD1C','FCER1'] # (biocompare)
#dccandi = sorted(list(set(dccandi) & set(dc_facs)))

# Treg
tregcandi = cellmarker_dic.get('Treg')
# generate original marker dict
k = ['B cells','T cells CD4','T cells CD8','NK cells','Monocytes','Neutrophils','Dendritic cells','Treg']
v = [bcandi,cd4candi,cd8candi,nkcandi,moncandi,neucandi,dccandi,tregcandi]
domain_dic = dict(zip(k,v))

pd.to_pickle(domain_dic,'/workspace/github/GLDADec/data/GSE107572/domain/gse107572_domain_dic.pkl')

#%% GSE60424
pre_domain_dic = pd.read_pickle('/workspace/github/GLDADec/data/GSE65133/domain/gse65133_domain_dic.pkl') # defined above section
cellmarker_dic = pd.read_pickle('/workspace/github/GLDADec/data/GSE107572/CellMarker/human_PBMC_CellMarker_8cell_raw_dic_230307.pkl')

# B cells
bcandi = []
bcandi.extend(pre_domain_dic.get('B cells memory'))
bcandi.extend(pre_domain_dic.get('B cells naive'))

# CD4 cells
cd4candi = []
cd4candi.extend(pre_domain_dic.get('T cells CD4 memory'))
cd4candi.extend(pre_domain_dic.get('T cells CD4 naive'))

# CD8 cels
cd8candi = pre_domain_dic.get('T cells CD8')

# NK cells
nkcandi = pre_domain_dic.get('NK cells')

# Lymphocyte
lymcandi = []
lymcandi.extend(bcandi)
lymcandi.extend(cd4candi)
lymcandi.extend(cd8candi)
lymcandi.extend(nkcandi)

# Monocytes
moncandi = pre_domain_dic.get('Monocytes')

# Neutrophils
neucandi = cellmarker_dic.get('Neutrophils')
neu_facs= ['CCR7','CD14','CD177','CD24','CD47','CD63','CD86','CXCR1','CXCR2','CXCR4','FCGR3A','FLT1','ICAM1','IL17RA',
           'ITGA4','ITGAM','ITGAX','ITGB2','PECAM1','PTPRC','SELL','SPN','TLR2','TLR4','TLR5','TLR7','TLR8','TLR9'] # (biocompare)
neucandi = sorted(list(set(neucandi) & set(neu_facs)))

# generate original marker dict
k = ['Lymphocytes','Monocytes','Neutrophils']
v = [lymcandi,moncandi,neucandi]
domain_dic = dict(zip(k,v))

pd.to_pickle(domain_dic,'/workspace/github/GLDADec/data/GSE60424/domain/gse60424_domain_dic.pkl')
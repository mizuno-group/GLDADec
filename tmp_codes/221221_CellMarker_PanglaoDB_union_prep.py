# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 14:27:41 2022

union of CellMarker and PanglaoDB

@author: I.Azuma
"""
import pandas as pd
import codecs
import collections
import matplotlib.pyplot as plt

#%% version 1 (CD8 marker are roughly selected)
# load each marker information (raw)
cellmarker_dic = pd.read_pickle('D:/GdriveSymbol/notebook/dry/Deconv_Method_dev/221201_method_dev_final_phase/221201_marker_prep/221201_human_marker/results/forGSE65133/human_PBMC_CellMarker_8cell_raw_dic.pkl')
""" marker candidates of CD8+ are wide """
panglao_dic = pd.read_pickle('D:/GdriveSymbol/notebook/dry/Deconv_Method_dev/221201_method_dev_final_phase/221201_marker_prep/221201_human_marker/results/forGSE65133/human_blood_Panglao_6cell_raw_dic.pkl')

#%% venn diagram comparison
from matplotlib_venn import venn2
def plot_venn2(data1:set,data2:set,title=None):
    """

    Parameters
    ----------
    data1 : set
        {'S100A12', 'CD1D', 'ASGR1', 'CFP', 'UPK3A', 'ASGR2', 'FCN1'}
    data2 : set
        {'S100A12', 'CD1D', 'FGL2', 'CD163', 'S100A6', 'IGKV1D-22', 'ASGR2', 'FCN1'}
    """
    # plot venn diagram
    plt.figure(figsize=(8,6))
    venn = venn2(subsets=[data1,data2],set_labels=('CellMarker','PanglaoDB'))
    plt.title(title)
    if len(data1&data2)>0:
        venn.get_label_by_id('110').set_text('\n'.join(map(str,data1&data2)))
        #venn.get_label_by_id('100').set_text('\n'.join(map(str,data1-data2)))
        #venn.get_label_by_id('010').set_text('\n'.join(map(str,data2-data1)))
    plt.show()

final_k = []
final_union = []
common_key = set(cellmarker_dic.keys()) & set(panglao_dic.keys())
for cell_name in common_key:
    data1 = set([t.upper() for t in cellmarker_dic.get(cell_name)])
    data2 = set([t.upper() for t in panglao_dic.get(cell_name)])
    final_union.append(sorted(list(set(data1) | set(data2))))
    final_k.append(cell_name)
    plot_venn2(data1, data2, title=str(cell_name))
# gamma delta
data1 = set([t.upper() for t in cellmarker_dic.get('T cells gamma delta')])
data2 = set([t.upper() for t in panglao_dic.get('Gamma delta T cells')])
final_union.append(sorted(list(set(data1) | set(data2))))
final_k.append('T cells gamma delta')
plot_venn2(data1, data2, title='Gamma delta T cells')
# CD4 naive
data1 = set([t.upper() for t in cellmarker_dic.get('T cells CD4 naive')])
final_union.append(sorted(list(set(data1))))
final_k.append('T cells CD4 naive')
# CD4 memory
data1 = set([t.upper() for t in cellmarker_dic.get('T cells CD4 memory')])
final_union.append(sorted(list(set(data1))))
final_k.append('T cells CD4 memory')
# CD8
data1 = set([t.upper() for t in cellmarker_dic.get('T cells CD8')])
final_union.append(sorted(list(set(data1))))
final_k.append('T cells CD8')

union_dic = dict(zip(final_k,final_union))
for i,k in enumerate(union_dic):
    print('---',k,'---')
    print(union_dic.get(k))
    print('')

pd.to_pickle(union_dic,'D:/GdriveSymbol/notebook/dry/Deconv_Method_dev/221201_method_dev_final_phase/221201_marker_prep/221201_human_marker/results/forGSE65133/human_PBMC_union_raw_dic_v1.pkl')

#%% version 2 (CD8 marker are strictly selected)
# load each marker information (raw)
cellmarker_dic = pd.read_pickle('D:/GdriveSymbol/notebook/dry/Deconv_Method_dev/221201_method_dev_final_phase/221201_marker_prep/221201_human_marker/results/forGSE65133_v2/human_PBMC_CellMarker_8cell_raw_dic_221216.pkl')
""" marker candidates of CD8+ are wide """
panglao_dic = pd.read_pickle('D:/GdriveSymbol/notebook/dry/Deconv_Method_dev/221201_method_dev_final_phase/221201_marker_prep/221201_human_marker/results/forGSE65133/human_blood_Panglao_6cell_raw_dic.pkl')

#%% venn diagram comparison
from matplotlib_venn import venn2
def plot_venn2(data1:set,data2:set,title=None):
    """

    Parameters
    ----------
    data1 : set
        {'S100A12', 'CD1D', 'ASGR1', 'CFP', 'UPK3A', 'ASGR2', 'FCN1'}
    data2 : set
        {'S100A12', 'CD1D', 'FGL2', 'CD163', 'S100A6', 'IGKV1D-22', 'ASGR2', 'FCN1'}
    """
    # plot venn diagram
    plt.figure(figsize=(8,6))
    venn = venn2(subsets=[data1,data2],set_labels=('CellMarker','PanglaoDB'))
    plt.title(title)
    if len(data1&data2)>0:
        venn.get_label_by_id('110').set_text('\n'.join(map(str,data1&data2)))
        #venn.get_label_by_id('100').set_text('\n'.join(map(str,data1-data2)))
        #venn.get_label_by_id('010').set_text('\n'.join(map(str,data2-data1)))
    plt.show()

final_k = []
final_union = []
common_key = set(cellmarker_dic.keys()) & set(panglao_dic.keys())
for cell_name in common_key:
    data1 = set([t.upper() for t in cellmarker_dic.get(cell_name)])
    data2 = set([t.upper() for t in panglao_dic.get(cell_name)])
    final_union.append(sorted(list(set(data1) | set(data2))))
    final_k.append(cell_name)
    plot_venn2(data1, data2, title=str(cell_name))
# gamma delta
data1 = set([t.upper() for t in cellmarker_dic.get('T cells gamma delta')])
data2 = set([t.upper() for t in panglao_dic.get('Gamma delta T cells')])
final_union.append(sorted(list(set(data1) | set(data2))))
final_k.append('T cells gamma delta')
plot_venn2(data1, data2, title='Gamma delta T cells')
# CD4 naive
data1 = set([t.upper() for t in cellmarker_dic.get('T cells CD4 naive')])
final_union.append(sorted(list(set(data1))))
final_k.append('T cells CD4 naive')
# CD4 memory
data1 = set([t.upper() for t in cellmarker_dic.get('T cells CD4 memory')])
final_union.append(sorted(list(set(data1))))
final_k.append('T cells CD4 memory')
# CD8
data1 = set([t.upper() for t in cellmarker_dic.get('T cells CD8')])
final_union.append(sorted(list(set(data1))))
final_k.append('T cells CD8')

union_dic = dict(zip(final_k,final_union))
for i,k in enumerate(union_dic):
    print('---',k,'---')
    print(union_dic.get(k))
    print('')

pd.to_pickle(union_dic,'D:/GdriveSymbol/notebook/dry/Deconv_Method_dev/221201_method_dev_final_phase/221201_marker_prep/221201_human_marker/results/forGSE65133_v2/human_PBMC_union_raw_dic_v2.pkl')
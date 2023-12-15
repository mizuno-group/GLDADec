# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 21:00:52 2021

@author: I.Azuma
"""

import numpy as np
import pandas as pd
import itertools
from sklearn.impute import SimpleImputer
import copy
from scipy.stats import rankdata
from tqdm import tqdm
from combat.pycombat import pycombat
import matplotlib.pyplot as plt

def annotation(df,ref_df, places:list=[0, 1]):
    """
    annotate row IDs to gene names
    Parameters
    ----------
    df : a dataframe to be analyzed
    ref_df : two rows of dataframe. e.g. ["Gene stable ID","MGI symbol"]
    places : list of positions of target rows in the ref_df
    """
    ref_df_dropna = ref_df.iloc[:,places].dropna(how='any', axis=0)
    id_lst = ref_df_dropna.iloc[:,0].tolist()
    symbol_lst = ref_df_dropna.iloc[:,1].tolist()
    conv_dict = dict(list(zip(id_lst, symbol_lst)))
    id_lst_raw = [str(x).split(".")[0] for x in df.index.tolist()] # ENSMUSG00000000049.12 --> ENSMUSG00000000049
    symbol_lst_new = [conv_dict.get(x, np.nan) for x in id_lst_raw]
    df_conv = copy.deepcopy(df)
    df_conv["symbol"] = symbol_lst_new # add new col
    df_conv = df_conv.dropna(subset=["symbol"])
    df_conv = df_conv.groupby("symbol").median() # take median value for duplication rows
    return df_conv

def annotation_legacy(df,ref_df):
    """
    annotate row IDs to gene names

    Parameters
    ----------
    df : a dataframe to be analyzed
    ref_df : two rows of dataframe. e.g. ["Gene stable ID","MGI symbol"]

    """
    ref_col = ref_df.columns.tolist()
    print("reference information :",ref_col)
    ids = ref_df[ref_col[0]].tolist()
    symbols = ref_df[ref_col[1]].tolist()
    
    total_id = [x.split(".")[0] for x in df.index.tolist()] # ENSMUSG00000000049.12 --> ENSMUSG00000000049
    total_res = [None]*len(df)
    
    id_set = set(ids)
    for i in tqdm(range(len(total_id))):
        if total_id[i] in id_set:
            j = ids.index(total_id[i])
            total_res[i] = symbols[j]
        else:
            pass
    
    total_df = copy.deepcopy(df)
    total_df["symbol"] = total_res # add new col
    drop_df = total_df.dropna() # remove rows which has no annotation
    group_df = drop_df.groupby("symbol").median() # take median value for duplication rows
    
    return group_df

def array_imputer(df,threshold=0.9,strategy="median",trim=1.0,batch=False,lst_batch=[], trim_red=True):
    """
    imputing nan and trim the values less than 1
    
    Parameters
    ----------
    df: a dataframe
        a dataframe to be analyzed
    
    threshold: float, default 0.9
        determine whether imupting is done or not dependent on ratio of not nan
        
    strategy: str, default median
        indicates which statistics is used for imputation
        candidates: "median", "most_frequent", "mean"
    
    lst_batch : lst, int
        indicates batch like : 0, 0, 1, 1, 1, 2, 2
    
    Returns
    ----------
    res: a dataframe
    
    """
    df_c = copy.deepcopy(df)
    if (type(trim)==float) or (type(trim)==int):
        df_c = df_c.where(df_c > trim)
    else:
        pass
    df_c = df_c.replace(0,np.nan)
    if batch:
        lst = []
        ap = lst.append
        for b in range(max(lst_batch)+1):
            place = [i for i, x in enumerate(lst_batch) if x == b]
            print("{0} ({1} sample)".format(b,len(place)))
            temp = df_c.iloc[:,place]
            if temp.shape[1]==1:
                ap(pd.DataFrame(temp))
            else:
                thresh = int(threshold*float(len(list(temp.columns))))
                temp = temp.dropna(thresh=thresh)
                imr = SimpleImputer(strategy=strategy)
                imputed = imr.fit_transform(temp.values.T) # impute in columns
                ap(pd.DataFrame(imputed.T,index=temp.index,columns=temp.columns))
        if trim_red:
            df_res = pd.concat(lst,axis=1)
            df_res = df_res.replace(np.nan,0) + 1
            print("redundancy trimming")
        else:
            df_res = pd.concat(lst,axis=1,join="inner")
    else:            
        thresh = int(threshold*float(len(list(df_c.columns))))
        df_c = df_c.dropna(thresh=thresh)
        imr = SimpleImputer(strategy=strategy)
        imputed = imr.fit_transform(df_c.values.T) # impute in columns
        df_res = pd.DataFrame(imputed.T,index=df_c.index,columns=df_c.columns)
    return df_res


def trimming(df, log=True, trimming=True, batch=False, lst_batch=[], trim_red=False, threshold=0.9):
    df_c = copy.deepcopy(df)
    # same index median
    df_c.index = [str(i) for i in df_c.index]
    df2 = pd.DataFrame()
    dup = df_c.index[df_c.index.duplicated(keep="first")]
    gene_list = pd.Series(dup).unique().tolist()
    if len(gene_list) != 0:
        for gene in gene_list:
            new = df_c.loc[:,gene].median()
            df2.loc[gene] = new
        df_c = df_c.drop(gene_list)
        df_c = pd.concat([df_c,df2.T])
    
    if trimming:
        if len(df_c.T) != 1:    
            df_c = array_imputer(df_c,lst_batch=lst_batch,batch=batch,trim_red=trim_red,threshold=threshold)
        else:
            df_c = df_c.where(df_c>1)
            df_c = df_c.dropna()
    else:
        df_c = df_c.dropna()

    # log conversion
    if log:
        df_c = df_c.where(df_c>=0)
        df_c = df_c.dropna()
        df_c = np.log2(df_c+1)
    else:
        pass
    return df_c

def batch_norm(df,lst_batch=[]):
    """
    batch normalization with combat
    
    Parameters
    ----------
    df: a dataframe
        a dataframe to be analyzed
    
    lst_batch : lst, int
        indicates batch like : 0, 0, 1, 1, 1, 2, 2
    
    """
    comb_df = pycombat(df,lst_batch)
    return comb_df

def multi_batch_norm(df,lst_lst_batch=[[],[]],do_plots=True):
    """
    batch normalization with combat for loop
    
    Note that the order of normalization is important. Begin with the broadest batch and move on to more specific batches of corrections.
    
    e.g. sex --> area --> country
    
    Parameters
    ----------
    df: a dataframe
        a dataframe to be analyzed
    
    lst_batch : lst, int
        indicates batch like : [[0,0,1,1,1,1],[0,0,1,1,2,2]]
    
    """
    df_c = df.copy() # deep copy
    for lst_batch in tqdm(lst_lst_batch):
        comb = batch_norm(df_c,lst_batch)
        df_c = comb # update
        if do_plots:
            for i in range(5):
                plt.hist(df_c.iloc[:,i],bins=200,alpha=0.8)
            plt.show()
        else:
            pass
    return df_c

def quantile(df,method="median"):
    """
    quantile normalization of dataframe (variable x sample)
    
    Parameters
    ----------
    df: dataframe
        a dataframe subjected to QN
    
    method: str, default "median"
        determine median or mean values are employed as the template    

    """
    #print("quantile normalization (QN)")
    df_c = df.copy() # deep copy
    lst_index = list(df_c.index)
    lst_col = list(df_c.columns)
    n_ind = len(lst_index)
    n_col = len(lst_col)

    ### prepare mean/median distribution
    x_sorted = np.sort(df_c.values,axis=0)[::-1]
    if method=="median":
        temp = np.median(x_sorted,axis=1)
    else:
        temp = np.mean(x_sorted,axis=1)
    temp_sorted = np.sort(temp)[::-1]

    ### prepare reference rank list
    x_rank_T = np.array([rankdata(v,method="ordinal") for v in df_c.T.values])

    ### conversion
    rank = sorted([v + 1 for v in range(n_ind)],reverse=True)
    converter = dict(list(zip(rank,temp_sorted)))
    converted = []
    converted_ap = converted.append  
    for i in range(n_col):
        transient = [converter[v] for v in list(x_rank_T[i])]
        converted_ap(transient)

    np_data = np.matrix(converted).T
    df2 = pd.DataFrame(np_data)
    df2.index = lst_index
    df2.columns = lst_col
    return df2

def log2(df):
    f_add = lambda x: x+1
    log_df = df.apply(f_add)
    log_df = np.log2(log_df)
    return log_df

def low_cut(df,threshold=1.0):
    df_c = copy.deepcopy(df)
    if (type(threshold)==float) or (type(threshold)==int):
        cut_df = df_c.where(df_c > threshold)
    else:
        pass
    return cut_df

def standardz_sample(x):
    pop_mean = x.mean(axis=0)
    pop_std = x.std(axis=0)+ np.spacing(1) # np.spacing(1) == np.finfo(np.float64).eps
    df = (x - pop_mean).divide(pop_std)
    df = df.replace(np.inf,np.nan)
    df = df.replace(-np.inf,np.nan)
    df = df.dropna()
    print('standardz population control')
    return df

def ctrl_norm(df,ctrl="C"):
    """normalization with ctrl samples"""
    ctrl_samples = []
    for t in df.index.tolist():
        if t.split("_")[0]==ctrl:
            ctrl_samples.append(t)
    ctrl_df = df.loc[ctrl_samples]
    
    ctrl_mean = ctrl_df.mean() # mean value of ctrl
    ctrl_std = ctrl_df.std() # std of ctrl
    
    norm_df = (df-ctrl_mean)/ctrl_std
    return norm_df

def drop_all_missing(df):
    replace = df.replace(0,np.nan)
    drop = replace.dropna(how="all") # remove rows whose all values are missing
    res = drop.fillna(0)
    print(len(df)-len(res),"rows are removed")
    return res

def freq_norm(df,marker_dic,ignore_others=True):
    """
    Normalize by sum of exression
    ----------
    df : DataFrame
        Genes in row and samples in column.
             PBMCs, 17-002  PBMCs, 17-006  ...  PBMCs, 17-060  PBMCs, 17-061
    AIF1          9.388634       8.354677  ...       8.848500       9.149019
    AIM2          4.675251       4.630904  ...       4.830909       4.831925
    ALOX5AP       9.064822       8.891569  ...       9.420134       9.192017
    APBA2         4.313265       4.455105  ...       4.309868       4.338142
    APEX1         7.581810       7.994079  ...       7.604995       7.706539
                   ...            ...  ...            ...            ...
    VCAN          8.213386       7.018457  ...       9.050750       8.263430
    VIPR1         6.436875       6.281543  ...       5.973437       6.622016
    ZBTB16        4.687727       4.618193  ...       4.730128       4.546280
    ZFP36        12.016052      11.514114  ...      11.538242      12.271717
    ZNF101        5.288079       5.250802  ...       5.029970       5.141903
    
    marker_dic : dict

    """
    others = sorted(list(set(df.index.tolist()) - set(itertools.chain.from_iterable(marker_dic.values()))))
    if len(others)>0:
        other_dic = {'others':others}
        #marker_dic = marker_dic | other_dic # Python 3.9
        marker_dic = {**marker_dic,**other_dic}

    # normalize
    use_k = []
    use_v = []
    for i,k in enumerate(marker_dic):
        if len(marker_dic.get(k))>0:
            use_k.append(k)
            use_v.append(marker_dic.get(k))
        else:
            pass
    marker_dic = dict(zip(use_k,use_v))
    
    cell_sums = []
    for i,k in enumerate(marker_dic):
        if ignore_others:
            if k == 'others':
                cell_sums.append(-1)
            else:
                common_v = sorted(list(set(marker_dic.get(k)) & set(df.index.tolist())))
                tmp_df = df.loc[common_v] # expression of markers
                tmp_sum = tmp_df.T.sum() # sum of expression level
                cell_sum = sum(tmp_sum)
                cell_sums.append(cell_sum)
        else:
            common_v = sorted(list(set(marker_dic.get(k)) & set(df.index.tolist())))
            tmp_df = df.loc[common_v] # expression of markers
            tmp_sum = tmp_df.T.sum() # sum of expression level
            cell_sum = sum(tmp_sum)
            cell_sums.append(cell_sum)
    
    base = max(cell_sums) # unify to maximum value
    r = [base/t for t in cell_sums]
    
    norm_df = pd.DataFrame()
    for i,k in enumerate(marker_dic):
        common_v = sorted(list(set(marker_dic.get(k)) & set(df.index.tolist())))
        tmp_df = df.loc[common_v] # expression of markers
        if ignore_others:
            if k == 'others':
                tmp_norm = tmp_df
            else:
                tmp_norm = tmp_df*r[i]
        else:
            tmp_norm = tmp_df*r[i]
        norm_df = pd.concat([norm_df,tmp_norm])
    
    # for multiple marker origin
    sample_name = norm_df.columns.tolist()[0]
    sort_norm = norm_df.sort_values(sample_name,ascending=False)
    trim_df = sort_norm[~sort_norm.index.duplicated(keep='first')]
    return trim_df


def size_norm(df,marker_dic):
    """
    Normalize by gene size (number).
    ----------
    df : DataFrame
        Genes in row and samples in column.
             PBMCs, 17-002  PBMCs, 17-006  ...  PBMCs, 17-060  PBMCs, 17-061
    AIF1          9.388634       8.354677  ...       8.848500       9.149019
    AIM2          4.675251       4.630904  ...       4.830909       4.831925
    ALOX5AP       9.064822       8.891569  ...       9.420134       9.192017
    APBA2         4.313265       4.455105  ...       4.309868       4.338142
    APEX1         7.581810       7.994079  ...       7.604995       7.706539
                   ...            ...  ...            ...            ...
    VCAN          8.213386       7.018457  ...       9.050750       8.263430
    VIPR1         6.436875       6.281543  ...       5.973437       6.622016
    ZBTB16        4.687727       4.618193  ...       4.730128       4.546280
    ZFP36        12.016052      11.514114  ...      11.538242      12.271717
    ZNF101        5.288079       5.250802  ...       5.029970       5.141903
    
    marker_dic : dict

    """
    max_size = max([len(t) for t in marker_dic.values()])
    norm_df = pd.DataFrame()
    for i,k in enumerate(marker_dic):
        common_v = sorted(list(set(marker_dic.get(k)) & set(df.index.tolist())))
        tmp_size = len(common_v)
        r = max_size / tmp_size
        tmp_df = df.loc[common_v] # expression of markers
        tmp_norm = tmp_df*r
        norm_df = pd.concat([norm_df,tmp_norm])
    return norm_df

def norm_total_res(total_res,base_names=['Monocytes', 'NK cells', 'B cells naive', 'B cells memory', 'T cells CD4 naive', 'T cells CD4 memory', 'T cells CD8', 'T cells gamma delta']):
    norm_total_res = []
    for tmp_df in total_res:
        tmp_df = tmp_df[base_names]
        tmp_sum = tmp_df.T.sum()
        r = 1/tmp_sum
        norm_res = (tmp_df.T*r).T
        norm_total_res.append(norm_res)
    return norm_total_res

def norm_val(val_df,base_names=['Naive B', 'Memory B', 'CD8 T', 'Naive CD4 T', 'Gamma delta T', 'NK', 'Monocytes']):
    tmp_df = val_df[base_names]
    tmp_sum = tmp_df.T.sum()
    r = 1/tmp_sum
    norm_res = (tmp_df.T*r).T
    return norm_res
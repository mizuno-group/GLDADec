from __future__ import absolute_import, unicode_literals  # noqa

import logging
import numbers
import sys

import numpy as np
import pandas as pd
import itertools

PY2 = sys.version_info[0] == 2
if PY2:
    import itertools
    zip = itertools.izip


logger = logging.getLogger('lda')


def check_random_state(seed):
    if seed is None:
        # i.e., use existing RandomState
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError("{} cannot be used as a random seed.".format(seed))


def matrix_to_lists(doc_word):
    """Convert a (sparse) matrix of counts into arrays of word and doc indices

    Parameters
    ----------
    doc_word : array or sparse matrix (D, V)
        document-term matrix of counts

    Returns
    -------
    (WS, DS) : tuple of two arrays
        WS[k] contains the kth word in the corpus
        DS[k] contains the document index for the kth word

    """
    """
    if np.count_nonzero(doc_word.sum(axis=1)) != doc_word.shape[0]:
        logger.warning("all zero row in document-term matrix found")
    if np.count_nonzero(doc_word.sum(axis=0)) != doc_word.shape[1]:
        logger.warning("all zero column in document-term matrix found")
    sparse = True
    """
    try:
        # if doc_word is a scipy sparse matrix
        doc_word = doc_word.copy().tolil()
    except AttributeError:
        sparse = False

    if sparse and not np.issubdtype(doc_word.dtype, np.integer):
        raise ValueError("expected sparse matrix with integer values, found float values")

    ii, jj = np.nonzero(doc_word)
    if sparse:
        ss = tuple(doc_word[i, j] for i, j in zip(ii, jj))
    else:
        ss = doc_word[ii, jj]

    DS = np.repeat(ii, ss).astype(np.intc)
    WS = np.repeat(jj, ss).astype(np.intc)
    return WS, DS


def lists_to_matrix(WS, DS):
    """Convert array of word (or topic) and document indices to doc-term array

    Parameters
    -----------
    (WS, DS) : tuple of two arrays
        WS[k] contains the kth word in the corpus
        DS[k] contains the document index for the kth word

    Returns
    -------
    doc_word : array (D, V)
        document-term array of counts

    """
    D = max(DS) + 1
    V = max(WS) + 1
    doc_word = np.zeros((D, V), dtype=np.intc)
    indices, counts = np.unique(list(zip(DS, WS)), axis=0, return_counts=True)
    doc_word[indices[:, 0], indices[:, 1]] += counts
    return doc_word


def dtm2ldac(dtm, offset=0):
    """Convert a document-term matrix into an LDA-C formatted file

    Parameters
    ----------
    dtm : array of shape N,V

    Returns
    -------
    doclines : iterable of LDA-C lines suitable for writing to file

    Notes
    -----
    If a format similar to SVMLight is desired, `offset` of 1 may be used.
    """
    try:
        dtm = dtm.tocsr()
    except AttributeError:
        pass
    assert np.issubdtype(dtm.dtype, np.integer)
    n_rows = dtm.shape[0]
    for i, row in enumerate(dtm):
        try:
            row = row.toarray().squeeze()
        except AttributeError:
            pass
        unique_terms = np.count_nonzero(row)
        if unique_terms == 0:
            raise ValueError("dtm row {} has all zero entries.".format(i))
        term_cnt_pairs = [(i + offset, cnt) for i, cnt in enumerate(row) if cnt > 0]
        docline = str(unique_terms) + ' '
        docline += ' '.join(["{}:{}".format(i, cnt) for i, cnt in term_cnt_pairs])
        if (i + 1) % 1000 == 0:
            logger.info("dtm2ldac: on row {} of {}".format(i + 1, n_rows))
        yield docline


def ldac2dtm(stream, offset=0):
    """Convert an LDA-C formatted file to a document-term array

    Parameters
    ----------
    stream: file object
        File yielding unicode strings in LDA-C format.

    Returns
    -------
    dtm : array of shape N,V

    Notes
    -----
    If a format similar to SVMLight is the source, an `offset` of 1 may be used.
    """
    doclines = stream

    # We need to figure out the dimensions of the dtm.
    N = 0
    V = -1
    data = []
    for l in doclines:  # noqa
        l = l.strip()  # noqa
        # skip empty lines
        if not l:
            continue
        unique_terms = int(l.split(' ')[0])
        term_cnt_pairs = [s.split(':') for s in l.split(' ')[1:]]
        for v, _ in term_cnt_pairs:
            # check that format is indeed LDA-C with the appropriate offset
            if int(v) == 0 and offset == 1:
                raise ValueError("Indexes in LDA-C are offset 1")
        term_cnt_pairs = tuple((int(v) - offset, int(cnt)) for v, cnt in term_cnt_pairs)
        np.testing.assert_equal(unique_terms, len(term_cnt_pairs))
        V = max(V, *[v for v, cnt in term_cnt_pairs])
        data.append(term_cnt_pairs)
        N += 1
    V = V + 1
    dtm = np.zeros((N, V), dtype=np.intc)
    for i, doc in enumerate(data):
        for v, cnt in doc:
            np.testing.assert_equal(dtm[i, v], 0)
            dtm[i, v] = cnt
    return dtm

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
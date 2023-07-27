# -*- coding: utf-8 -*-
"""
Created on 2023-05-23 (Tue) 15:54:13

data preprocessing

@author: I.Azuma
"""
#%%
import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import sys
from pathlib import Path
BASE_DIR = Path(__file__).parent

from _utils import processing

from logging import getLogger
logger = getLogger('dev0')

#%%
class PreProcessing():
    def __init__(self):
        self.mix_raw = None
    
    def set_data(self,mix_raw,ann_ref=None,batch_info=None):
        self.mix_raw = mix_raw
        self.ann_ref = ann_ref
        self.batch_info = batch_info
        logger.info('original: {}'.format(self.mix_raw.shape))
    
    def sample_selection(self,target_samples=['Ctrl','APAP']):
        if len(target_samples)>0:
            samples = self.mix_raw.columns.tolist()
            use_samples = []
            for t in samples:
                if t.split('_')[0] in target_samples:
                    use_samples.append(t)
                else:
                    pass
            self.target_df = self.mix_raw[use_samples]
        else:
            self.target_df = self.mix_raw
        logger.info('sample selection: {}'.format(self.target_df.shape))

            
    def preprocessing(self,do_ann=True,linear2log=False,log2linear=False,do_drop=True,do_batch_norm=True,do_quantile=True):
        # gene name duplication
        if len(self.target_df) != len(set(self.target_df.index.tolist())):
            tmp_df = copy.deepcopy(self.target_df)
            idx_name = tmp_df.index.tolist()
            tmp_df['symbol'] = idx_name
            tmp_df = tmp_df.dropna(subset=["symbol"])
            self.target_df = tmp_df.groupby("symbol").median() # take median value for duplication rows

        # annotation
        if do_ann:
            self.target_df = processing.annotation(self.target_df, self.ann_ref)
            logger.info('annotation: {}'.format(self.target_df.shape))
        else:
            pass
        # linear --> log2
        if linear2log:
            df_c = copy.deepcopy(self.target_df)
            self.target_df = processing.log2(df_c)
            logger.info('linear2log: {}'.format(self.target_df.shape))
        else:
            pass
        # log2 --> linear
        if log2linear:
            df_c = copy.deepcopy(self.target_df)
            fxn = lambda x : 2**x if x < 30 else 1073741824 # FIXME: avoid overflow
            self.target_df = df_c.applymap(fxn)
            logger.info('log2linear: {}'.format(self.target_df.shape))
        else:
            pass
        # trimming
        if do_drop:
            df_c = copy.deepcopy(self.target_df)
            self.target_df = df_c.replace(0,np.nan).dropna(how='all').replace(np.nan,0)
            logger.info('trimming: {}'.format(self.target_df.shape))
        else:
            pass
        # batch normalization
        if do_batch_norm:
            df_c = copy.deepcopy(self.target_df)
            info = self.batch_info.loc[df_c.columns.tolist()] # sample selection

            replace_list = info["replace"].tolist()
            prep_list = info["prep_batch"].tolist()
            lane_list = info["lane_batch"].tolist()
            lst_batch = [replace_list,prep_list,lane_list]

            comb_df = processing.multi_batch_norm(df_c,lst_batch,do_plots=False)
            fxn = lambda x : 0 if x<0 else x
            self.target_df = comb_df.applymap(fxn) # negative expression level is not acceptable
            logger.info('batch normalization: {}'.format(self.target_df.shape))
        else:
            pass
        # quantile normalization
        if do_quantile:
            df_c = copy.deepcopy(self.target_df)
            qn_df = processing.quantile(df_c)
            fxn = lambda x : 0 if x<0 else x
            self.target_df = qn_df.applymap(fxn)
            logger.info('quantile normalization: {}'.format(self.target_df.shape))
        else:
            pass


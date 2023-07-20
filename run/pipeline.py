# -*- coding: utf-8 -*-
"""
Created on 2023-05-09 (Tue) 10:00:47

Pipeline for comprehensive analysis

@author: I.Azuma
"""
#%%
import gc
import copy
import random
import pprint
import logging
import itertools
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.preprocessing import MinMaxScaler

Base_dir = '/workspace/github/GLDADec' # cloning repository
import sys
sys.path.append(Base_dir)

from run import dev0_preprocessing
from run import dev1_set_data
from run import dev2_anchor_detection
from run import dev3_deconvolution
from run import dev4_evaluation
from gldadec import utils
from _utils import processing, plot_utils

#%%
logger = logging.getLogger('pipeline')

#%%
class Pipeline():
    def __init__(self,verbose=False):
        self.df = None
        self.marker_dic = None
        self.verbose = verbose
        self.mm_df = None
    
    def from_predata(self,mix_raw,ann_ref=None,batch_info=None,target_samples=['Ctrl','APAP'],
                    do_ann=True,log2linear=False,linear2log=False,do_drop=True,do_batch_norm=True,do_quantile=True,remove_noise=False):
        """
        You can skip below 1.set_data() and 2.sample_selection()
        """
        # TODO: log2linear may lead to OverflowError ('Numerical result out of range'). Monitor and warn of scale.
        PP = dev0_preprocessing.PreProcessing()
        PP.set_data(mix_raw=mix_raw,ann_ref=ann_ref,batch_info=batch_info)
        PP.sample_selection(target_samples=target_samples)
        PP.preprocessing(do_ann=do_ann,log2linear=log2linear,linear2log=linear2log,do_drop=do_drop,do_batch_norm=do_batch_norm,do_quantile=do_quantile)
        target_df = PP.target_df
        target_df.index = [t.upper() for t in target_df.index.tolist()]
        if remove_noise:
            # remove ribosomal and mitochondrial genes
            rps = [] # ribosomal protein
            mts = [] # mitochondrial gene
            target_genes = []
            for g in target_df.index.tolist():
                if g[0:3] in ['RPL','RPS']:
                    rps.append(g)
                elif g[0:3] in ['MT-']:
                    mts.append(g)
                else:
                    target_genes.append(g)
            target_df = target_df.loc[sorted(target_genes)]
            logger.info('ribosomal_genes: {}'.format(len(rps)))
            logger.info('mitochondrial_genes: {}'.format(len(mts)))
        else:
            pass
        self.target_df = target_df

    def set_data(self,df,marker_dic:dict):
        df.index = [t.upper() for t in df.index.tolist()]
        upper_v = []
        for i,k in enumerate(marker_dic):
            upper_v.append([t.upper() for t in marker_dic.get(k)])
        new_dic = dict(zip(list(marker_dic.keys()), upper_v))
        self.df = df
        self.marker_dic = new_dic

    def sample_selection(self,target_samples=['Ctrl', 'APAP']):
        """
        Ctrl vs one administration group
        """
        if len(target_samples)==0:
            use_samples = self.df.columns.tolist()
        else:
            samples = self.df.columns.tolist()
            use_samples = []
            for t in samples:
                if t.split('_')[0] in target_samples:
                    use_samples.append(t)
                else:
                    pass
        self.target_df = self.df[use_samples]
        logger.info('n_samples: {}'.format(len(use_samples)))
    
    def gene_selection(self,method='CV',outlier=True,topn=500):
        """
        select target genes other than marker genes
        """
        target_df = copy.deepcopy(self.target_df)
        if method=='CV':
            if outlier:
                PP = dev0_preprocessing.PreProcessing()
                log_df = processing.log2(target_df)
                common = set(log_df.index.tolist())
                for sample in log_df.columns.tolist():
                    log_sample = log_df[sample].replace(0,np.nan).dropna()
                    mu = log_sample.mean()
                    sigma = log_sample.std()
                    df3 = log_sample[(mu - 2*sigma <= log_sample) & (log_sample <= mu + 2*sigma)]
                    common = common & set(df3.index.tolist())
                target_df = target_df.loc[sorted(list(common))]
            else:
                pass
            var_df = pd.DataFrame(target_df.T.var())
            mean_df = pd.DataFrame(target_df.T.mean())
            cv_df = var_df/mean_df # coefficient of variance
            cv_df = cv_df.sort_values(0,ascending=False)

            # top n
            self.high_genes = cv_df.index.tolist()[0:topn]
            self.high_df = cv_df.loc[self.high_genes]
        else:
            raise ValueError('!! set other method !!')
        logger.info('method: {}, outlier:{}, n_top: {}'.format(method,outlier,topn))
    
    def add_marker_genes(self,target_cells=['B', 'NK', 'Neutrophil', 'Monocyte', 'Eosinophil', 'Basophil', 'Kupffer'],add_dic=None):
        if add_dic is None:
            marker_dic = self.marker_dic
        else:
            upper_v = []
            for i,k in enumerate(add_dic):
                upper_v.append([t.upper() for t in add_dic.get(k)])
            marker_dic = dict(zip(list(add_dic.keys()), upper_v))
            
        if len(target_cells)==0:
            target_cells = list(marker_dic.keys())
        target_v = []
        for cell in target_cells:
            target_v.append(marker_dic.get(cell))
        self.target_dic = dict(zip(target_cells,target_v))

        # gene definition
        marker_genes = list(itertools.chain.from_iterable(list(self.target_dic.values()))) # marker genes
        common_marker = sorted(list(set(marker_genes) & set(self.target_df.index.tolist()))) # marker genes that are registered
        target_genes = sorted(list(set(common_marker) | set(self.high_genes))) # 500 + 189

        self.added_df = self.target_df.loc[target_genes]

        # other genes
        self.other_genes = sorted(list(set(self.added_df.index) - set(itertools.chain.from_iterable(self.target_dic.values()))))
        logger.info('target_cells: {}, n_genes: {}'.format(target_cells,len(target_genes)))

    # FIXME: below 'prior_info_norm' method is included in 'deconv_prep' method.
    def prior_info_norm(self,scale=1000,norm=True):
        """
        Allowing duplication of marker genes does not work well.
        """
        if norm:
            linear_norm = utils.freq_norm(self.added_df,self.target_dic)
            linear_norm = linear_norm.loc[sorted(linear_norm.index.tolist())]
            self.deconv_df = linear_norm/scale
        else:
            self.deconv_df = self.added_df/scale
        logger.info('prior_norm: {}'.format(norm))
    
    def deocnv_prep(self,random_sets:list,do_plot=True,specific=True,prior_norm=True,norm_scale=1000,minmax=True,mm_scale=10):
        # dev1
        SD = dev1_set_data.SetData(verbose=self.verbose)
        SD.set_expression(df=self.added_df) 
        SD.set_marker(marker_dic=self.target_dic)
        SD.marker_info_processing(do_plot=do_plot)
        SD.set_random(random_sets=random_sets)
        SD.expression_processing(random_genes=self.other_genes,random_n=0,specific=specific,prior_norm=prior_norm,norm_scale=norm_scale)
        SD.seed_processing()

        # Collect data to be used in later analyses
        self.input_mat = SD.input_mat
        self.final_int = SD.final_int
        self.seed_topics = SD.seed_topics
        self.marker_final_dic = SD.marker_final_dic

        # correlation between samples
        if do_plot:
            cor = self.final_int.corr()
            sns.heatmap(cor)
            plt.show()
        if minmax:
            # Sample-wide normalization
            mm_scaler = MinMaxScaler()
            self.mm_df = (pd.DataFrame(mm_scaler.fit_transform(self.final_int.T),index=self.final_int.T.index, columns=self.final_int.T.columns)*mm_scale).T
            # correlation between samples
            if do_plot:
                cor = self.mm_df.corr()
                sns.heatmap(cor)
                plt.show()
            logger.info('minmax_scaling: {}'.format(mm_scale))
        else:
            pass
        gc.collect()
    
    def deconv(self,n=100,add_topic=0,n_iter=100,alpha=0.01,eta=0.01,refresh=10,initial_conf=1.0,seed_conf=1.0,other_conf=0.0,ll_plot=True,var_plot=True):
        if self.mm_df is None:
            deconv_df = self.final_int
        else:
            deconv_df = self.mm_df
        original_order = sorted(deconv_df.index.tolist())
        merge_total_res = []
        gene_contribution_res = []
        final_ll_list = []
        for i in tqdm(range(n)):
            random.seed(i+1)
            re_order = random.sample(original_order,len(original_order)) # randomly sort the gene order
            mm_target = deconv_df.loc[re_order]
            # conduct deconvolution
            # dev3
            Dec = dev3_deconvolution.Deconvolution(verbose=False)
            Dec.set_marker(marker_final_dic=self.marker_final_dic,anchor_dic=self.marker_final_dic)
            Dec.marker_redefine()
            Dec.set_random(random_sets=[123])
            Dec.set_final_int(final_int=mm_target) # sample-wide norm and re-order

            Dec.seed_processing()
            seed_topics = Dec.seed_topics
            Dec.ensemble_deconv(add_topic=add_topic,n_iter=n_iter,alpha=alpha,eta=eta,refresh=refresh,initial_conf=initial_conf,seed_conf=seed_conf,other_conf=other_conf)

            # log-likelihood
            ll_list = Dec.ll_list
            final_ll_list.append(ll_list[-1])
            logger.info("<iter: {}> log-likelihood: {}".format(i+1, ll_list[-1]))
            if ll_plot:
                if i == 0:
                    fig,ax = plt.subplots(figsize=(5,4),dpi=100)
                    for idx in range(len(ll_list)):
                        x = [(i+1)*10 for i in range(len(ll_list[idx]))]
                        plt.plot(x,ll_list[idx],label="seed: "+str(123)) # now using random state 123
                    plt.xlabel('iterations')
                    plt.ylabel('log-likelihood')
                    plt.gca().spines['right'].set_visible(False)
                    plt.gca().spines['top'].set_visible(False)
                    plt.gca().yaxis.set_ticks_position('left')
                    plt.gca().xaxis.set_ticks_position('bottom')
                    ax.set_axisbelow(True)
                    plt.show()
            total_res = Dec.total_res
            merge_total_res.append(total_res[0])
            gene_contribution_res.append(Dec.gene_contribution)
            gc.collect()
        self.merge_total_res = merge_total_res
        self.gene_contribution_res = gene_contribution_res
        self.final_ll_list = final_ll_list
        # var plot
        if var_plot:
            cell_candi = total_res[0].columns.tolist()
            for cell in cell_candi:
                try:
                    plot_utils.estimation_var(total_res=merge_total_res,cell=str(cell))
                except:
                    pass
        logger.info('n_ensemble: {}, n_add_topics: {}, n_iter: {}'.format(n,add_topic,n_iter))
        gc.collect()
            
    def evaluate(self,facs_df=None,deconv_norm_range=['NK','Neutrophil','Monocyte','Eosinophil','Kupffer'],facs_norm_range=['NK','Monocyte','Neutrophil','Kupffer','Eosinophil'],
    res_names=[['Neutrophil'],['Monocyte'],['NK'],['Eosinophil'],['Kupffer']],
    ref_names=[['Neutrophil'],['Monocyte'],['NK'],['Eosinophil'],['Kupffer']],
    title_list=['NK','Neutrophil','Monocyte','Eosinophil','Kupffer'],
    target_samples = None,
    figsize=(6,6),dpi=50,plot_size=100,multi=True):
        """
        ----------
        ref_df : pd.DataFrame
        cells in rows and samples in columns
        
                       Naive B  Memory B  CD8 T  ...  Gamma delta T     NK  Monocytes
        PBMCs, 17-002    12.36      2.96  22.36  ...           5.44  16.58      17.10
        PBMCs, 17-006     3.33      1.83  32.05  ...           2.28   8.90       5.70
        PBMCs, 17-019    15.09      4.07  17.86  ...          12.49  29.64       7.99
        PBMCs, 17-023    16.77      2.95   8.69  ...           0.64  17.11      17.08
        PBMCs, 17-026     8.40      4.69  27.98  ...           5.59   7.88      13.38
        PBMCs, 17-027    11.62      4.16  20.37  ...           7.55  14.18      10.47

        """
        Eval = dev4_evaluation.Evaluation()
        # normalize
        if len(deconv_norm_range)==0:
            self.norm_res = self.merge_total_res
        else:
            self.norm_res = utils.norm_total_res(self.merge_total_res,base_names=deconv_norm_range)
        
        if len(facs_norm_range)==0:
            norm_facs = facs_df
        else:
            norm_facs = utils.norm_val(val_df=facs_df,base_names=facs_norm_range)
        
        # evaluation
        Eval.set_res(total_res=self.norm_res,z_norm=False)
        Eval.set_ref(ref_df=norm_facs,z_norm=False)
        self.ensemble_res = Eval.ensemble_res
        if multi:
            Eval.multi_eval_multi_group(res_names=res_names,ref_names=ref_names,title_list=title_list,figsize=figsize,dpi=dpi,plot_size=plot_size) # visualization
        else:
            Eval.multi_eval(res_names=res_names,ref_names=ref_names,title_list=title_list,target_samples=target_samples,figsize=figsize,dpi=dpi,plot_size=plot_size)

        self.performance_dic = Eval.performance_dic
        pprint.pprint(self.performance_dic)
        logger.info('deconv_norm_range: {}'.format(deconv_norm_range))
        logger.info('facs_norm_range: {}'.format(facs_norm_range))
        logger.info('res_names: {}'.format(res_names))
        logger.info('facs_names: {}'.format(ref_names))
    
    def add_profile_eval(self,add_topic=10,topn=None,alternative='less',do_plot=True):
        if alternative not in ['less','greater']:
            raise ValueError('!! Inappropriate alternative setting !!')

        gcr = self.gene_contribution_res
        # summarize gene contributions
        pvalue_list = []
        cor_list = []
        max_p_list = []
        min_p_list = []
        overlap = False
        for t in gcr:
            sorted_genes = sorted(t[0].index.tolist())
            gc_df = t[0].loc[sorted_genes]
            add_gc = gc_df[[i+1 for i in range(add_topic)]] # added topics
            other_genes = self.other_genes
            add_gc_other = add_gc.loc[other_genes] # added gene contribution to added topics
            if topn is None:
                topn = int(len(add_gc_other)/add_topic) # soft threshold

            target_genes = []
            for t in add_gc_other.columns.tolist():
                tmp_df = add_gc_other[[t]].sort_values(t,ascending=False)
                top_genes = tmp_df.index.tolist()[0:topn] # high contribution to the topic
                target_genes.extend(top_genes)
            target_gc_other = add_gc_other.loc[sorted(list(set(target_genes)))]

            # overlap eval
            if len(target_gc_other) != len(target_genes):
                overlap = True
            # metric
            cor = target_gc_other.corr()
            cor_list.append(cor)
            pval = target_gc_other.corr(method=lambda x, y: pearsonr(x, y,alternative=alternative)[1])
            pvalue_list.append(pval)
            max_pval = np.triu(np.array(pval),k=1).max()
            tmp =  np.triu(np.array(pval),k=1)
            min_pval =np.where(tmp==0,1,tmp).min() # no considerations other than the upper triangle
            max_p_list.append(max_pval)
            min_p_list.append(min_pval)

        # select
        if alternative == 'less':
            #target_p = max(max_p_list) # max(max_p_list) is the most strict condition
            target_p = min(max_p_list) # min(max_p_list) is mild condition
            target_index = max_p_list.index(target_p)
        else:
            target_p = min(min_p_list)
            target_index = min_p_list.index(target_p)

        p_res = pvalue_list[target_index]
        cor_res = cor_list[target_index]

        if do_plot:
            # correlation eval
            fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharex=True, sharey=True)
            sns.heatmap(cor_res,ax=axes[0],annot=True,fmt="1.2f")

            # pvalue eval
            sns.heatmap(p_res,ax=axes[1],annot=True,fmt="1.1e",cmap='cividis',annot_kws={"fontsize":6})
            plt.show()
        if target_p < 0.05:
            pval_flag = False
        else:
            pval_flag = True

        logger.info('overlap: {}, pvalue: {}'.format(overlap,pval_flag))
        return overlap, pval_flag, min_p_list, max_p_list

    def add_profile_eval_legacy(self,add_topic=10,topn=None):
        gcr = self.gene_contribution_res
        # summarize gene contributions
        # FIXME: Each additional topic may not be equivalent
        l = []
        for t in gcr:
            sorted_genes = sorted(t[0].index.tolist())
            gc_df = t[0].loc[sorted_genes]
            l.append(gc_df)
        gc_m = sum(l)/len(l)
        add_gc = gc_m[[i+1 for i in range(add_topic)]] # added topics
        other_genes = self.other_genes
        add_gc_other = add_gc.loc[other_genes] # added gene contribution to added topics
        if topn is None:
            topn = int(len(add_gc_other)/add_topic/10) # soft threshold

        target_genes = []
        for t in add_gc_other.columns.tolist():
            tmp_df = add_gc_other[[t]].sort_values(t,ascending=False)
            top_genes = tmp_df.index.tolist()[0:topn] # high contribution to the topic
            target_genes.extend(top_genes)
        target_gc_other = add_gc_other.loc[sorted(list(set(target_genes)))]

        # overlap eval
        if len(target_gc_other) != len(target_genes):
            overlap = True
        else:
            overlap = False
        
        # correlation eval
        fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharex=True, sharey=True)
        cor = target_gc_other.corr()
        sns.heatmap(cor,ax=axes[0],annot=True,fmt="1.2f")

        cor = cor.replace(1,-1)
        cor_max = cor.max().max()
        if cor_max > 0:
            posi_cor = True
        else:
            posi_cor = False

        # pvalue eval
        pval = target_gc_other.corr(method=lambda x, y: pearsonr(x, y,alternative='greater')[1])
        sns.heatmap(pval,ax=axes[1],annot=True,fmt="1.1e",cmap='cividis',annot_kws={"fontsize":6})
        plt.show()

        min_pvalue = pval.min().min()
        if min_pvalue < 0.05:
            posi_pvalue = True
        else:
            posi_pvalue = False

        logger.info('overlap: {}, positive_correlation: {}'.format(overlap,posi_cor))
        return overlap, posi_pvalue

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
import itertools
import pandas as pd
import seaborn as sns
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

Base_dir = '/workspace/github/GLDADec' # cloning repository

import sys
sys.path.append(Base_dir)
from Dev import dev_utils
from Dev import dev1_set_data
from Dev import dev2_anchor_detection
from Dev import dev3_deconvolution
from Dev import dev4_evaluation

from Dev.gldadec import utils

#%%
class Pipeline():
    def __init__(self,verbose=False):
        self.df = None
        self.marker_dic = None
        self.verbose=verbose
    
    def set_data(self,df,marker_dic:dict):
        self.df = df
        self.marker_dic = marker_dic
    
    def sample_selection(self,target_samples=['Ctrl', 'APAP']):
        """
        Ctrl vs one administration group
        """
        samples = self.df.columns.tolist()
        use_samples = []
        for t in samples:
            if t.split('_')[0] in target_samples:
                use_samples.append(t)
            else:
                pass
        self.target_df = self.df[use_samples]
    
    def gene_selection(self,method='CV',topn=500):
        """
        select target genes other than marker genes
        """
        target_df = copy.deepcopy(self.target_df)
        if method=='CV':
            var_df = pd.DataFrame(target_df.T.var())
            mean_df = pd.DataFrame(target_df.T.mean())
            cv_df = var_df/mean_df # coefficient of variance
            cv_df = cv_df.sort_values(0,ascending=False)

            # top n
            self.high_genes = cv_df.index.tolist()[0:topn]
            self.high_df = cv_df.loc[self.high_genes]
        else:
            raise ValueError('!! set other method !!')
    
    def add_marker_genes(self,target_cells = ['B', 'NK', 'Neutrophil', 'Monocyte', 'Eosinophil', 'Basophil', 'Kupffer']):
        marker_dic = self.marker_dic
        target_v = []
        for cell in target_cells:
            target_v.append(marker_dic.get(cell))
        self.target_dic = dict(zip(target_cells,target_v))

        # gene definition
        marker_genes = list(itertools.chain.from_iterable(list(self.target_dic.values()))) # marker genes
        common_marker = sorted(list(set(marker_genes) & set(self.df.index.tolist()))) # marker genes that are registered
        target_genes = sorted(list(set(common_marker) | set(self.high_genes))) # 500 + 189

        self.target_linear = self.target_df.loc[target_genes]

        # other genes
        self.other_genes = sorted(list(set(self.target_linear.index) - set(itertools.chain.from_iterable(self.target_dic.values()))))
    
    def prior_info_norm(self,scale=1000,norm=True):
        if norm:
            linear_norm = utils.freq_norm(self.target_linear,self.target_dic)
            linear_norm = linear_norm.loc[sorted(linear_norm.index.tolist())]
            self.deconv_df = linear_norm/scale
        else:
            self.deconv_df = self.target_linear
    
    def deocnv_prep(self,random_sets:list,do_plot=True,specific=True,scale=10):
        # dev1
        SD = dev1_set_data.SetData(verbose=self.verbose)
        SD.set_expression(df=self.deconv_df) 
        SD.set_marker(marker_dic=self.target_dic)
        SD.marker_info_processing(do_plot=do_plot)
        SD.set_random(random_sets=random_sets)
        SD.expression_processing(random_genes=self.other_genes,random_n=0,specific=specific)
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

        # Sample-wide normalization
        mm_scaler = MinMaxScaler()
        self.mm_df = (pd.DataFrame(mm_scaler.fit_transform(self.final_int.T),index=self.final_int.T.index, columns=self.final_int.T.columns)*scale).T

        # correlation between samples
        if do_plot:
            cor = self.mm_df.corr()
            sns.heatmap(cor)
            plt.show()
    
    def deconv(self,n=100,add_topic=0,n_iter=100,alpha=0.01,eta=0.01,refresh=10,initial_conf=1.0,seed_conf=1.0,other_conf=0.0,ll_plot=True,var_plot=True):
        original_order = sorted(self.mm_df.index.tolist())
        merge_total_res = []
        gene_contribution_res = []
        final_ll_list = []
        for i in tqdm(range(n)):
            random.seed(i+1)
            re_order = random.sample(original_order,len(original_order)) # randomly sort the gene order

            mm_target = self.mm_df.loc[re_order]
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
                    dev_utils.estimation_var(total_res=merge_total_res,cell=str(cell))
                except:
                    pass
            
    def evaluate(self,facs_df=None,deconv_norm_range=['NK','Neutrophil','Monocyte','Eosinophil','Kupffer'],facs_norm_range=['NK','Monocyte','Neutrophil','Kupffer','Eosinophil'],
    res_names=[['Neutrophil'],['Monocyte'],['NK'],['Eosinophil'],['Kupffer']],
    ref_names=[['Neutrophil'],['Monocyte'],['NK'],['Eosinophil'],['Kupffer']],dpi=50):
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
            norm_res = self.merge_total_res
        else:
            norm_res = utils.norm_total_res(self.merge_total_res,base_names=deconv_norm_range)
        
        if len(facs_norm_range)==0:
            norm_facs = facs_df
        else:
            norm_facs = utils.norm_val(val_df=facs_df,base_names=facs_norm_range)
        
        # evaluation
        Eval.set_res(total_res=norm_res,z_norm=False)
        Eval.set_ref(ref_df=norm_facs,z_norm=False)
        self.ensemble_res = Eval.ensemble_res
        Eval.multi_eval_multi_group(res_names=res_names,ref_names=ref_names,dpi=dpi) # visualization

        self.cor_dic = Eval.cor_dic
        pprint.pprint(self.cor_dic)

    def add_profile_eval(self,add_topic=10,topn=None):
        gcr = self.gene_contribution_res
        # summarize gene contributions
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
        cor = target_gc_other.corr()
        sns.heatmap(cor,annot=True,fmt="1.2f")
        plt.show()

        cor = cor.replace(1,-1)
        cor_max = cor.max().max()
        if cor_max > 0:
            posi_cor = True
        else:
            posi_cor = False
        return overlap,posi_cor

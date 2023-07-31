# -*- coding: utf-8 -*-
"""
Created on 2023-07-04 (Tue) 17:15:02

Additional Topic Eval

@author: I.Azuma
"""
#%%
import copy
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import fcluster

import sys
sys.path.append('/workspace/github/enan')
from enan import FET

#%%
class AddTopicEval():
    def __init__(self,add_topic=6,target_cells=['Kupffer cell', 'Natural killer cell', 'Monocyte', 'Neutrophil']):
        self.add_topic = add_topic
        self.target_cells = target_cells
        self.cluster_df = None
    
    def set_gene_contr_res(self,gene_contr_res:list):
        """ set gene contribution list

        Args:
            gene_contr_res (list): pipeline.Pipeline().gene_contribution_res
        """
        self.gene_contr_res = gene_contr_res
    
    def set_deconv_res(self,deconv_res:list):
        """ set deconvolution output

        Args:
            deconv_res (list): pipeline.Pipeline().merge_total_res
        """
        self.deocnv_res = deconv_res
    
    def flatten(self):
        """
        additional_summary: ()
        """
        gene_contr_res = self.gene_contr_res
        additional_summary = pd.DataFrame()
        for idx in range(len(gene_contr_res)):
            each_res = gene_contr_res[idx][0]
            additional_contr = each_res.drop(columns=self.target_cells)
            additional_summary = pd.concat([additional_summary,additional_contr],axis=1)

        additional_summary.columns = [i+1 for i in range(len(additional_summary.T))]

        # gene-wide normalization
        mm_scaler = MinMaxScaler()
        mm_df = (pd.DataFrame(mm_scaler.fit_transform(additional_summary),index=additional_summary.index, columns=additional_summary.columns))

        #sns.clustermap(mm_df,z_score=1)
        #plt.show()

        self.additional_summary = mm_df
    
    def topic_clustering(self,method='kmeans',do_plot=True,n_clusters=None):
        if method not in ['kmeans','hierarchical']:
            raise ValueError('!! Inappropriate Method !!')
        
        if n_clusters is None:
            n_clusters = self.add_topic
        self.n_clusters = n_clusters
        
        if method=='kmeans':
            kmeans = KMeans(n_clusters=self.n_clusters)
            kmeans.fit(self.additional_summary.T)
            label_info = kmeans.labels_
        else:
            z = linkage(self.additional_summary.T,metric='euclidean',method='average')
            label_info = [l-1 for l in list(fcluster(z, t=self.n_clusters, criterion='maxclust'))]
        
        cluster_df = copy.deepcopy(self.additional_summary)
        cluster_df.columns = label_info
        
        if do_plot:
            sns.clustermap(cluster_df,z_score=1)
        
        self.cluster_df = cluster_df

    def conduct_fet(self,ref_dic:dict,threshold=None,cluster=True):
        # pd.read_pickle('/workspace/github/enan/enan/enrichr/KEGG_2019_Mouse_ref_dic.pkl')
        if cluster:
            if self.cluster_df is None:
                raise ValueError("!! conduct topic_clustering or change to cluster=False !!")
            else:
                num = self.n_clusters
        else:
            self.cluster_df = copy.deepcopy(self.additional_summary)
            self.cluster_df.columns = [i for i in range(len(self.cluster_df.T))]
            num = len(self.cluster_df.T)

        if threshold is None:
            threshold = 1/self.n_clusters
        res_summary = []
        for i in range(num):
            try:
                contr_df = pd.DataFrame(self.cluster_df[i].mean(axis=1))
            except:
                contr_df = pd.DataFrame(self.cluster_df[i])
                contr_df.columns = [0]
            
            high_gene = set(contr_df[contr_df[0]>threshold].index.tolist())
            dat = FET() # generate an instance
            dat.fit(ref_dic) # load reference
            res = dat.calc(high_gene) # analyze data of interest
            res_summary.append(res)
        self.res_summary = res_summary
    
    



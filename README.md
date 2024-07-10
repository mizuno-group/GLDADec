# GLDADec: Guided LDA Deconvolution
Deconvolution is a computational method that can be applied to estimate the proportion of immune cells in the sample from the transcriptome data.
Here, we proposed a Guided LDA Deconvolution method, called GLDADec, to estimate cell type proportions by using marker gene names as partial prior information.

<img src="https://github.com/mizuno-group/LiverDeconv/assets/92911852/93c7c6dc-ec6d-4471-824d-23efecd38e75" width=800>

## Publication
- [peer-reviewed publication](https://academic.oup.com/bib/article/25/4/bbae315/7709575)  
- [preprint](https://www.biorxiv.org/content/10.1101/2024.01.08.574749v2)  

## Getting Started
#### Step 1. Clone this repository
```
git clone https://github.com/mizuno-group/LiverDeconv.git
```

#### Step 2. Build for Cython files.
```
pip install cython
cd ./GLDADec/gldadec
python setup.py build_ext --inplace
```
※ If an error ```'gcc' failed: No such file or directory``` appears, perform  ```sudo apt-get install gcc```.

#### Step 3. Running
GLDADec mainly inputs the bulk gene expression profiles to be analyzed and the marker gene names for each cell type. The processed data and the code for processing are in the ```./data/``` folder.

``` Python
BASE_DIR = '/workspace/github/GLDADec'  # path to the cloned repository

import pandas as pd
import sys
sys.path.append(BASE_DIR)
from run import pipeline

raw_df = pd.read_csv(BASE_DIR+'/data/GSE65133/GSE65133_expression.csv',index_col=0)  # bulk gene expression
domain_dic = pd.read_pickle(BASE_DIR+'/data/marker/human_blood_domain.pkl')  # marker gene names for each cell type
target_facs = pd.read_csv(BASE_DIR+'/data/GSE65133/facs.csv',index_col=0)/100  # true values measured by FACS
random_sets = pd.read_pickle(BASE_DIR+'/data/random_info/100_random_sets.pkl')

# single run and eval
pp = pipeline.Pipeline(verbose=False)
pp.from_predata(raw_df,target_samples=[],
                do_ann=False,linear2log=False,log2linear=False,do_drop=True,
                do_batch_norm=False,do_quantile=False,remove_noise=False)
pp.gene_selection(method='CV',outlier=True,topn=100)
pp.add_marker_genes(target_cells=['Monocytes','NK cells','B cells naive','B cells memory',
                                  'T cells CD4 naive','T cells CD4 memory','T cells CD8','T cells gamma delta'],
                    add_dic=domain_dic)
pp.deconv_prep(random_sets=random_sets,do_plot=False,specific=True,prior_norm=True,norm_scale=10,minmax=False,mm_scale=10)
pp.deconv(n=10,add_topic=0,n_iter=100,alpha=0.01,eta=0.01,refresh=10,
          initial_conf=1.0,seed_conf=1.0,other_conf=0.0,ll_plot=True,var_plot=False)

# evaluate
pp.evaluate(facs_df=target_facs,
    deconv_norm_range=['Monocytes', 'NK cells', 'B cells naive', 'B cells memory',
                       'T cells CD4 naive', 'T cells CD4 memory', 'T cells CD8', 'T cells gamma delta'],
    facs_norm_range=[],
    res_names=[['B cells naive'],['B cells memory'],['T cells CD4 naive'],['T cells CD4 memory'],
               ['T cells CD8'],['NK cells'],['Monocytes'],['T cells gamma delta']],
    ref_names=[['Naive B'],['Memory B'],['Naive CD4 T'],['Resting memory CD4 T', 'Activated memory CD4 T'],
              ['CD8 T'],['NK'],['Monocytes'],['Gamma delta T']],
    title_list = ['Naive B','Memory B','Naive CD4 T','Memory CD4 T','CD8 T','NK','Monocytes','Gamma delta T'],
    figsize=(6,6),dpi=50,plot_size=100,multi=False)

# output
merge_res = pp.merge_total_res
deconv_res = sum(merge_res) / len(merge_res)  # ensemble
```
By referring to the sample codes in  ```./example/``` folder, you can perform a more detailed analysis or reproduce a paper using GLDADec.

## Authors
- [Iori Azuma](https://github.com/groovy-phazuma)  
    - main contributor  
- [Tadahaya Mizuno](https://github.com/tadahayamiz)  
    - correspondence  

## Contact
If you have any questions or comments, please feel free to create an issue on github here, or email us:  
- phazuma19980625[at]gmail.com  
- tadahaya[at]gmail.com  
- tadahaya[at]mol.f.u-tokyo.ac.jp  

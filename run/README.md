## Running

`pipeline.py` integrates several classes described below. You can perform the following tasks by manipulating this file.

1. Input expression data, sample selection, and preprocessing.
2. Selection of genes with high variability for analysis.
3. Input marker genes to be used as prior information.
4. Condition setting for deconvolution.
5. Performing GLDADec.
6. Evaluate by comparing the estimated value with the true value measured by flow cytometry.

***
#### Components of `pipeline.py`
- `dev0_preprocessing.py`: Data preprocessing class.

- `dev1_set_data.py`: Loading and setting data class.
    1. Input raw gene expression data.
    2. Set marker genes information.
    3. Refine marker genes according to the target data.
    4. Final processing of expression data.
    5. Preparation of prior information for topic guiding.
       
- `dev2_deconvolution.py`: Deconvolution core class

- `dev3_evaluation.py`: Performance evaluation class.

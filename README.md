

# Inter-domain Multi-relational Link Prediction
This is the codes for the paper [Inter-domain Multi-relational Link Prediction](https://arxiv.org/abs/2106.06171), ECML-PKDD 2021


# Requirements
torchkge==0.16.17 \
optuna=2.5.0

Complete environment setting can be found in './conda_environment.yml'


# Data
Download data from https://drive.google.com/file/d/1vyBPkLmLdunJrZoCds5n9_K_gTrmiU_V/view?usp=sharing
and unzip to 'data/' directory

# To run the codes
Go to './Rescal/' directory

## Prepare warmstart embeddings with Rescal

1. From scratch
Run 'rescal_torch/create_warmstart_embeddings_*.py' to create warmstart embeddings for each dataset \
Change the output path to create warmstart embeddings for hyper-search and for main experiments (note that this separation between hyper-search and main experiments is just hitorical design-choice.)


2. The ones used in the paper
Download zip file from \
https://drive.google.com/file/d/10UI7R9-qsJee68PXMn8FiH70bO3nVMQ7/view?usp=sharing \
and \
https://drive.google.com/file/d/1ccWBIcI77hxnqURpLzaFPhWea1yBGT1O/view?usp=sharing \

then unzip to 'rescal_torch/hyper-search_warmstart_embeddings/' and 'rescal_torch/warmstart_embeddings/' directories


## Tuning hyper-parameters

Run 'rescal_torch/run_hyper_search_*.sh' for tuning hyper-parameters in each dataset \
Or \
Use the ones provided in 'rescal_torch/best_hypers/' directory 


## Running training codes

Run 'rescal_torch/run_*.sh' to train RESCAL/ the proposed method + WD/ the proposed method + MMD


## Results

Hit@10 scores are automatically saved to txt files in 'rescal_torch/fix-intra_semi_results/' after training finishes.

ROC-AUC scores are saved to txt files in 'rescal_torch/fix-intra_semi_AUCs/' by running 'rescal_torch/compute_AUCs_*.py'


## Learned embeddings

The learned embeddings after training are saved in 'rescal_torch/fix-intra_semi_embeddings/'

The ones learned from the paper's experiments can be downloaded from https://drive.google.com/file/d/1UJ4GdxXGHdcYRjCrj5agXD7KHEBzVvFR/view?usp=sharing


## Note
In the paper, a mixture of Gaussian kernels with fixed bandwidths is used as the kernel for MMD. Empirically, the mixture performs favourably to a single Gaussian kernel with a tuned bandwidth. \
For results of MMD with tuned bandwidth, check './tuned_bandwidth_MMD_rescal_torch/' directory.


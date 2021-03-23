

# Requirements
torchkge==0.16.17 \
optuna=2.5.0

Complete environment setting can be found in '../conda_environment.yml'


# Data
Download data from https://drive.google.com/file/d/1vyBPkLmLdunJrZoCds5n9_K_gTrmiU_V/view?usp=sharing
and unzip to 'data/' directory

# To run the codes

## Prepare warmstart embeddings with Rescal

1. From scratch
Run 'torch_rescal/create_warmstart_embeddings_*.py' to create warmstart embeddings for each dataset


2. The ones used in the paper
Download zip file from \
https://drive.google.com/file/d/10UI7R9-qsJee68PXMn8FiH70bO3nVMQ7/view?usp=sharing \
and \
https://drive.google.com/file/d/1ccWBIcI77hxnqURpLzaFPhWea1yBGT1O/view?usp=sharing \

then unzip to 'torch_rescal/hyper-search_warmstart_embeddings/' and 'torch_rescal/warmstart_embeddings/' directories


## Tuning hyper-parameters

Run 'torch_rescal/run_hyper_search_*.sh' for tuning hyper-parameters in each dataset \
Or \
Use the ones provided in 'torch_rescal/best_hypers/' directory 


## Running training codes

Run 'torch_rescal/run_*.sh' to train RESCAL/ the proposed method + WD/ the proposed method + MMD

## Results

Hit@10 scores are automatically saved in 'torch_rescal/fix-intra_semi_results/' after training finishes.

ROC-AUC scores are saved in 'torch_rescal/fix-intra_semi_AUCs/' by running 'torch_rescal/compute_AUCs_*.py'


## Learned embeddings

The learned embeddings after training are saved in 'torch_rescal/fix-intra_semi_embeddings/'

The ones learned from the paper's experiments can be downloaded from https://drive.google.com/file/d/1UJ4GdxXGHdcYRjCrj5agXD7KHEBzVvFR/view?usp=sharing





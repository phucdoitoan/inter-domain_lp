

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from semi_mul_ot_rescal import Semi_Rescal

from utils import UniformNegativeSampler_Extended as Negative_Sampler
#from utils import UniformNegativeSampler as Negative_Sampler  # do not use normal uniform_negative sampler as it will assign small score to cross-domain triplets -> not good for cross-domain prediction
#from torchkge.sampling import UniformNegativeSampler as Negative_Sampler

from utils import Extended_KnowledgeGraph as KnowledgeGraph

from tqdm import tqdm
from time import time

import pandas as pd

import matplotlib.pyplot as plt

import sys

from torchkge.utils.operations import get_dictionaries
from torchkge.utils import MarginLoss

from torchkge.evaluation import LinkPredictionEvaluator

import pickle

from utils import load_embedding, sample_intra_negative_triplets, sample_inter_negative_triplets

import optuna

import random

# hyper_space
#EPOCHS = range(100, 500)

#batch_size = [100, 500, 1000]  # could not be so large

#learning_rate = [0.005, 0.001, 0.01, 0.0005]

#emb_dim = [50, 100, 200] # not so large

#alpha = [0.5, 1.0, 3.0, 5.0, 7.0, 10.0, 20.0, 50.0]



def objective(trial, data_file, data_name, overlap):

	# fix hyper
	MARGIN = 1.0
	LAMDA = 1.0
	inter_data_prop = 0.0

	# tuning hyper
	alpha = trial.suggest_float('alpha', 0.5, 10.0, log=True)

	#emb_dim = trial.suggest_categorical('emb_dim', [50, 100, 150])
	emb_dim = trial.suggest_categorical('emb_dim', [100])

	#EPOCHS = trial.suggest_categorical('EPOCHS', [200, 300, 400, 500])
	EPOCHS = trial.suggest_categorical('EPOCHS', [300])

	#batch_size = trial.suggest_categorical('batch_size', [100, 300, 500, 700])
	#batch_size = trial.suggest_categorical('batch_size', [100, 300])
	batch_size = trial.suggest_categorical('batch_size', [300])
	
	#learning_rate = trial.suggest_categorical('lr', [0.01, 0.005, 0.001, 0.0005])
	learning_rate = trial.suggest_categorical('lr', [0.01, 0.005, 0.001])

	#bandwith mu for gaussian kernel in MMD
	mu = trial.suggest_float('mu', 0.01, 2.0, log=True)
	if mu > 1.0:
		mu = 10 * (mu - 0.9) # if mu > 1.0, scale mu to (1.0, 11.)
	# mu is uniformly sampled from (0.01, 1.0) and (1.0, 11.)


	print('trial alpha: ', alpha)
	print('trial emb_dim: ', emb_dim)
	print('trial EPOCHS: ', EPOCHS)
	print('trial batch_size: ', batch_size)
	print('trial learning_rate: ', learning_rate)

	# other fix hypers
	verbose = False

	device = torch.device('cuda:%s' %torch.cuda.current_device() if torch.cuda.is_available() else 'cpu')
	print('device: ', device)

	# laod data 
	with open(data_file, 'rb') as file:
		kg_dict = pickle.load(file)

	kg = kg_dict['intra_train']
	#intra_kg_test = kg_dict['intra_test']
	inter_kg_valid = kg_dict['inter_valid']
	#inter_kg_test = kg_dict['inter_test']
	n1 = kg_dict['n1']
	n2 = kg_dict['n2']
	n_common = kg_dict['n_common']

	print('n1: %s - n_common: %s - n2: %s' %(n1, n_common, n2))
	print('entities overlapped level: %.4f' %(n_common/(n1+n_common)))
	print('Intra train triplets: ', kg.n_facts)
	#print('Intra test triplets: ', intra_kg_test.n_facts)
	print('Inter valid triplets: ', inter_kg_valid.n_facts)
	#print('Inter test triplets: ', inter_kg_test.n_facts)


	# averaging over 3 times

	average_hit10 = []

	#warmstart_list = random.sample(list(range(5)), 3)
	warmstart_list = list(range(5))

	for repeat in warmstart_list:
		ent_file_name = 'hyper-search_warmstart_embeddings/%s_WARM-START_overlap=%s_ent_embedding_Emb-dim=%s_Batch-size=500_lr=0.005_EPOCHS=100_repeat=%s.pt' %(data_name, overlap, emb_dim, repeat)
		rel_file_name = 'hyper-search_warmstart_embeddings/%s_WARM-START_overlap=%s_rel_embedding_Emb-dim=%s_Batch-size=500_lr=0.005_EPOCHS=100_repeat=%s.pt' %(data_name, overlap, emb_dim, repeat)


		print('load warmstart ent embeddings from: ', ent_file_name)
		print('               rel embeddings from: ', rel_file_name)

		init_ent_emb = torch.load(ent_file_name)
		init_rel_emb = torch.load(rel_file_name)

		#Model
		semi_model = Semi_Rescal(emb_dim=emb_dim, n_ent_list=[n1, n2, n_common], n_rel=kg.n_rel, alpha=alpha, device=device, \
								 init_ent_emb=init_ent_emb, init_rel_emb=init_rel_emb, \
								 use_MMD=True, MMD_kernel_mu=mu, \
								 eps=1e-4, lamda=LAMDA, max_iter=100, thresh=1e-9, w_max_iter=100, w_thresh=1e-9,\
								 stable_sinkhorn=True, data_precision='float', verbose=verbose)  #maybe we need to tune eps as well, not so necessary though

		# criterion
		criterion = MarginLoss(1.0)

		# dataloader
		dataloader = DataLoader(kg, batch_size=batch_size, shuffle=False)

		# negative sampler
		negative_sampler = Negative_Sampler(kg, n1=semi_model.n1, n_common=semi_model.n_common, n2=semi_model.n2)

		# optimizer
		optimizer = torch.optim.Adam(semi_model.parameters(), lr=learning_rate, weight_decay=1e-5)


		best_hit10 = 0

		patience_budgest = 50
		patience_count = 0
		epoch_num_to_check_hit10 = 5

		EPOCHS_iter = tqdm(range(EPOCHS), unit='epoch')
		for epoch in EPOCHS_iter:
			
			total_batch = 0
			running_loss = 0.0

			for h, t, r in dataloader:
				total_batch += 1

				h, t, r = h.to(device), t.to(device), r.to(device)

				n_h, n_t = negative_sampler.corrupt_batch(h, t, r)  # TODO: quiz, can it be used in the combined model?????

				optimizer.zero_grad()
				pos, neg, pos_regul, neg_regul, ot_loss = semi_model(h, t, n_h, n_t, r)

				loss = criterion(pos, neg) + (pos_regul + neg_regul)/2 + ot_loss*alpha

				loss.backward()
				optimizer.step()

				running_loss += loss.item()

			# normalize parameters
			semi_model.normalize_parameters()

			ot_cost = semi_model.update_P()  # if alpha = 0.0 => do nothing, reurn torch.tensor(0.0)

			# compute Hit@10 scores after each epoch
			if epoch % epoch_num_to_check_hit10 == 0:
				semi_model.is_intra=False  # need to set is_intra to False before evaluate
				evaluator_inter = LinkPredictionEvaluator(semi_model, inter_kg_valid)
				evaluator_inter.evaluate(1000, 10, verbose=False)
				hit10 = evaluator_inter.hit_at_k(k=10)[1]

				if hit10 > best_hit10:
					best_hit10 = hit10
					patience_count = 0
				else:
					patience_count += epoch_num_to_check_hit10

				# early stopping
				if patience_count > patience_budgest:
					print('\nStop with EARLY STOPPING: break at repeat %s - epoch %s' %(repeat, epoch))
					average_hit10.append(best_hit10)
					break

			EPOCHS_iter.set_description('Ep: %s | mean loss: %.5f | ot loss: %.5f | Best Hit@10: %.4f | Patience: %s' %(epoch, running_loss/total_batch, ot_cost.item(), best_hit10, patience_count))

	average_hit10 = torch.tensor(average_hit10)
	average_hit10 = average_hit10.mean()

		## report intermediate values to optuna
		#trial.report(best_hit10, epoch)

		# handle pruning based on the intermediate value
		#if trial.should_prune():
		#	raise optuna.exceptions.TrialPruned()

	return average_hit10


data_file_dict = {
			'FB15k-237': [('../data/FB15k-237/divided/fix_intra-test/kg_valid_dict.pkl', 0.0),
						  ('../data/FB15k-237/semi/divided/fix_intra-test/kg_valid_dict.pkl', 1.5),
						  ('../data/FB15k-237/semi_3/divided/fix_intra-test/kg_valid_dict.pkl', 3.0),
						  ('../data/FB15k-237/semi_5/divided/fix_intra-test/kg_valid_dict.pkl', 5.0), ],
			'WN18RR': [('../data/WN18RR/divided/fix_intra-test/kg_valid_dict.pkl', 0.0),
						  ('../data/WN18RR/semi/divided/fix_intra-test/kg_valid_dict.pkl', 1.5),
						  ('../data/WN18RR/semi_3/divided/fix_intra-test/kg_valid_dict.pkl', 3.0),
						  ('../data/WN18RR/semi_5/divided/fix_intra-test/kg_valid_dict.pkl', 5.0), ],
			'DBbook2014': [('../data/KG_datasets/dbbook2014/kg/divided/fix_intra-test/kg_valid_dict.pkl', 0.0),
						  ('../data/KG_datasets/dbbook2014/semi/divided/fix_intra-test/kg_valid_dict.pkl', 1.5),
						  ('../data/KG_datasets/dbbook2014/semi_3/divided/fix_intra-test/kg_valid_dict.pkl', 3.0),
						  ('../data/KG_datasets/dbbook2014/semi_5/divided/fix_intra-test/kg_valid_dict.pkl', 5.0), ],
			'ML1M': [('../data/KG_datasets/ml1m/kg/divided/fix_intra-test/kg_valid_dict.pkl', 0.0),
						  ('../data/KG_datasets/ml1m/semi/divided/fix_intra-test/kg_valid_dict.pkl', 1.5),
						  ('../data/KG_datasets/ml1m/semi_3/divided/fix_intra-test/kg_valid_dict.pkl', 3.0),
						  ('../data/KG_datasets/ml1m/semi_5/divided/fix_intra-test/kg_valid_dict.pkl', 5.0), ],
		}







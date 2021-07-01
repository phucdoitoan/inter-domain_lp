

from sklearn.metrics import roc_auc_score


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

import pickle


def load_embedding(filename):
	
	emb = torch.load(filename)
	print('############################ Warm-start! Loaded embedidng from %s #######################' % filename)
	return emb


def filter_neg_triplet_from_sampled_triplets(true_triplets, sam_triplets):
	true_set = {true_triplets[:,i].reshape(-1,1) for i in range(true_triplets.shape[1])}
	sam_set = {sam_triplets[:,i].reshape(-1,1) for i in range(sam_triplets.shape[1])}

	true_neg_set = sam_set.difference(true_set)

	neg_triplets = torch.cat(list(true_neg_set), dim=1)

	return neg_triplets


def sample_intra_negative_triplets(true_triplets, neg_num, n_rel, n1, n2, n_common):

	neg_num1 = int(neg_num/2)
	neg_num2 = neg_num - neg_num1

	neg_heads1 = torch.randint(0, n1+n_common, (1,neg_num1))
	neg_tails1 = torch.randint(0, n1+n_common, (1,neg_num1))
	neg_heads2 = torch.randint(n1, n1+n_common+n2, (1,neg_num2))
	neg_tails2 = torch.randint(n1, n1+n_common+n2, (1,neg_num2))

	neg_heads = torch.cat((neg_heads1, neg_heads2), dim=1)
	neg_tails = torch.cat((neg_tails1, neg_tails2), dim=1)

	neg_rels = torch.randint(0, n_rel, (1,neg_num))

	neg_triplets = torch.cat((neg_heads, neg_rels, neg_tails), dim=0)

	neg_triplets = filter_neg_triplet_from_sampled_triplets(true_triplets, neg_triplets)

	if neg_triplets.shape[1] < neg_num:
		additional_neg_triplets = sample_intra_negative_triplets(true_triplets, neg_num - neg_triplets.shape[1], n_rel, n1, n1, n_common)
		neg_triplets = torch.cat((neg_triplets, additional_neg_triplets), dim=1)

	return neg_triplets


def sample_inter_negative_triplets(true_triplets, neg_num, n_rel, n1, n2, n_common):

	neg_num1 = int(neg_num/2)
	neg_num2 = neg_num - neg_num1

	neg_heads1 = torch.randint(0, n1, (1,neg_num1))
	neg_tails1 = torch.randint(n1+n_common, n1+n_common+n2, (1,neg_num1))
	neg_heads2 = torch.randint(n1+n_common, n1+n_common+n2, (1,neg_num2))
	neg_tails2 = torch.randint(0, n1, (1,neg_num2))

	neg_heads = torch.cat((neg_heads1, neg_heads2), dim=1)
	neg_tails = torch.cat((neg_tails1, neg_tails2), dim=1)

	neg_rels = torch.randint(0, n_rel, (1, neg_num))

	neg_triplets = torch.cat((neg_heads, neg_rels, neg_tails), dim=0)

	neg_triplets = filter_neg_triplet_from_sampled_triplets(true_triplets, neg_triplets)

	if neg_triplets.shape[1] < neg_num:
		additional_neg_triplets = sample_intra_negative_triplets(true_triplets, neg_num - neg_triplets.shape[1], n_rel, n1, n1, n_common)
		neg_triplets = torch.cat((neg_triplets, additional_neg_triplets), dim=1)

	return neg_triplets


def index_tensor_from_kgs(kg):
	heads = kg.head_idx.reshape(1,-1)
	rels = kg.relations.reshape(1,-1)
	tails = kg.tail_idx.reshape(1,-1)

	return torch.cat((heads, rels, tails), dim=0)


EMB_DIM = 100

LAMDA = 1.0

@torch.no_grad()
def compute_AUC(alpha, data_path, data_name, overlap, repeat_index=0):

	path = data_path

	ALPHA = alpha

	inter_data_prop = 0.

	print('inter data prop: ', inter_data_prop)
	print('alpha: ', ALPHA)
	print('lamda: ', LAMDA)

	verbose = False #True #False

	device = torch.device('cuda:%s' %torch.cuda.current_device() if torch.cuda.is_available() else 'cpu')
	print('device: ', device)

	# Load data
	with open(path + 'kg_valid_dict.pkl', 'rb') as file:
		kg_dict = pickle.load(file)

	kg = kg_dict['intra_train']
	intra_kg_test = kg_dict['intra_test']
	inter_kg_test = kg_dict['inter_test']
	n1 = kg_dict['n1']
	n2 = kg_dict['n2']
	n_common = kg_dict['n_common']

	print('n1: %s - n_common: %s - n2: %s' %(n1, n_common, n2))
	print('entities overlapped level: %.4f' %(n_common/(n1+n_common)))
	print('Intra train triplets: ', kg.n_facts)
	print('Intra test triplets: ', intra_kg_test.n_facts)
	print('Inter test triplets: ', inter_kg_test.n_facts)


	# Load embedding learned with alpha=0. WARM START
	#if alpha == 5.0:
	#	ent_file_name = 'fix-intra_semi_embeddings/' + '%s_WARM-START_SEMI%s_all_ent_embedding_alpha=%s_lamda=%s_semi=%s_repeat=%s.pt' %(data_name, overlap, 5.0, LAMDA, inter_data_prop, repeat_index)
	#	rel_file_name = 'fix-intra_semi_embeddings/' + '%s_WARM-START_SEMI%s_all_rel_embedding_alpha=%s_lamda=%s_semi=%s_repeat=%s.pt' %(data_name, overlap, 5.0, LAMDA, inter_data_prop, repeat_index)
	#elif alpha == 0.0:
	#	ent_file_name = 'fix-intra_semi_embeddings/' + '%s_WARM-START_SEMI%s_warmstart0_all_ent_embedding_alpha=%s_lamda=%s_semi=%s_repeat=%s.pt' %(data_name, overlap, 0.0, LAMDA, inter_data_prop, repeat_index)
	#	rel_file_name = 'fix-intra_semi_embeddings/' + '%s_WARM-START_SEMI%s_warmstart0_all_rel_embedding_alpha=%s_lamda=%s_semi=%s_repeat=%s.pt' %(data_name, overlap, 0.0, LAMDA, inter_data_prop, repeat_index)
	#else:
	#	raise Exception('Wrong values for alpha: %s' %alpha)

	ent_file_name = 'fix-intra_semi_embeddings/' + '%s_WARM-START_SEMI%s_all_ent_embedding_alpha=%s_lamda=%s_semi=%s_repeat=%s.pt' %(data_name, overlap, ALPHA, LAMDA, inter_data_prop, repeat_index)
	rel_file_name = 'fix-intra_semi_embeddings/' + '%s_WARM-START_SEMI%s_all_rel_embedding_alpha=%s_lamda=%s_semi=%s_repeat=%s.pt' %(data_name, overlap, ALPHA, LAMDA, inter_data_prop, repeat_index)

	init_ent_emb = load_embedding(ent_file_name)
	init_rel_emb = load_embedding(rel_file_name)

	# MODEL
	semi_model = Semi_Rescal(emb_dim=EMB_DIM, n_ent_list=[n1, n2, n_common], n_rel=kg.n_rel, alpha=ALPHA, device=device,\
							 init_ent_emb=init_ent_emb, init_rel_emb=init_rel_emb, \
							 eps=1e-4, lamda=LAMDA, max_iter=100, thresh=1e-9, w_max_iter=100, w_thresh=1e-9,\
							 stable_sinkhorn=True, data_precision='float', verbose=verbose)


	# intra test loader
	intra_train_true_triplets = index_tensor_from_kgs(kg).to(device)
	intra_test_true_triplets = index_tensor_from_kgs(intra_kg_test).to(device)

	intra_test_neg_triplets = sample_intra_negative_triplets(torch.cat((intra_train_true_triplets, intra_test_true_triplets), dim=1), \
									intra_test_true_triplets.shape[1], kg.n_rel, n1, n2, n_common).to(device)

	intra_labels = torch.cat((torch.ones_like(intra_test_true_triplets[0]), torch.zeros_like(intra_test_neg_triplets[0]))).cpu().numpy()

	intra_test_true_scores = semi_model.scoring_function(intra_test_true_triplets[0], intra_test_true_triplets[2], intra_test_true_triplets[1])
	intra_test_neg_scores = semi_model.scoring_function(intra_test_neg_triplets[0], intra_test_neg_triplets[2], intra_test_neg_triplets[1])

	intra_scores = torch.cat((intra_test_true_scores, intra_test_neg_scores)).cpu().numpy()

	print('intra labels: ', intra_labels.shape)
	print('intra scores: ', intra_scores.shape)

	intra_roc = roc_auc_score(intra_labels, intra_scores)

	print('******* intra ROC AUC: %.4f ************' %intra_roc)



	# inter test 
	inter_test_true_triplets = index_tensor_from_kgs(inter_kg_test).to(device)

	inter_test_neg_triplets = sample_inter_negative_triplets(inter_test_true_triplets, inter_test_true_triplets.shape[1], kg.n_rel, n1, n2, n_common).to(device)

	inter_labels = torch.cat((torch.ones_like(inter_test_true_triplets[0]), torch.zeros_like(inter_test_neg_triplets[0]))).cpu().numpy()

	inter_test_true_scores = semi_model.scoring_function(inter_test_true_triplets[0], inter_test_true_triplets[2], inter_test_true_triplets[1])
	inter_test_neg_scores = semi_model.scoring_function(inter_test_neg_triplets[0], inter_test_neg_triplets[2], inter_test_neg_triplets[1])

	inter_scores = torch.cat((inter_test_true_scores, inter_test_neg_scores)).cpu().numpy()

	print('inter labels: ', inter_labels.shape)
	print('inter scores: ', inter_scores.shape)

	inter_roc = roc_auc_score(inter_labels, inter_scores)

	print('*********** inter ROC AUC: %.4f *************** ' %inter_roc)


	return intra_roc, inter_roc


repeats = 10

if __name__ == '__main__':

	path_dict = {
	'-UNOVERLAPED': '../data/%s/divided/fix_intra-test/' ,
	'' : '../data/%s/semi/divided/fix_intra-test/' ,
	'-3': '../data/%s/semi_3/divided/fix_intra-test/' ,
	'-5': '../data/%s/semi_5/divided/fix_intra-test/' ,
	}

	overlap_dict = {
		'-UNOVERLAPED': '0.0',
		'': '1.5',
		'-3': '3.0',
		'-5': '5.0',
	}

	with torch.cuda.device(0):

		data_name = 'FB15k-237'

		with open('fix-intra_semi_AUCs/%s_AUCs.txt' %(data_name), 'w') as file:
			for overlap in ['-UNOVERLAPED', '', '-3', '-5']:
				overlap_percent = overlap_dict[overlap]

				for alpha in [0., 5.]:

					if alpha != 0:
						with open('best_hypers/%s_overlap=%s_best_hypers.pkl' %(data_name, overlap_percent), 'rb') as hyper_file:
							best_hypers = pickle.load(hyper_file).params
						
						alpha = best_hypers['alpha']


					data_path = path_dict[overlap] %data_name

					intra_rocs, inter_rocs = [], []

					for repeat_index in range(repeats):
						intra_roc, inter_roc = compute_AUC(alpha, data_path, data_name, overlap=overlap, repeat_index=repeat_index)

						intra_rocs.append(intra_roc)
						inter_rocs.append(inter_roc)

					intra_rocs = torch.tensor(intra_rocs)
					inter_rocs = torch.tensor(inter_rocs)

					print('Average intra ROC AUC: %.4f (+-%.4f)' %(intra_rocs.mean(), intra_rocs.std()))
					print('Average inter ROC AUC: %.4f (+-%.4f)' %(inter_rocs.mean(), inter_rocs.std()))

					
					file.write('\n**********************\n')
					file.write('Data name: %s\n' %(data_name))
					file.write('Alpha: %s\n' %alpha)
					file.write('Overlap: %s\n' %overlap)
					file.write('Average intra ROC AUC: %.4f (+-%.4f)\n' %(intra_rocs.mean(), intra_rocs.std()))
					file.write('Average inter ROC AUC: %.4f (+-%.4f)\n' %(inter_rocs.mean(), inter_rocs.std()))





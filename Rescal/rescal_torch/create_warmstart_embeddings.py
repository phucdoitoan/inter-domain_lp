

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from semi_mul_ot_rescal import Semi_Rescal

from utils import UniformNegativeSampler_Extended as Negative_Sampler

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


def main(data_file, data_name, emb_dim, batch_size, learning_rate, overlap, repeat):

	EPOCHS = 100

	ALPHA = 0.0

	LAMDA = 1.0

	print('******** create warmstart embeddings for Hyper Search *************')
	print('\ndata_file: ', data_file)
	print('data name: ', data_name)
	print('emb_dim: ', emb_dim)
	print('batch_size: ', batch_size)
	print('learning_rate: ', learning_rate)

	verbose = False
	device = torch.device('cuda:%s' %torch.cuda.current_device() if torch.cuda.is_available() else 'cpu')
	print('device: ', device)

	# laod data
	with open(data_file, 'rb') as file:
		kg_dict = pickle.load(file)

	kg = kg_dict['intra_train']
	n1 = kg_dict['n1']
	n2 = kg_dict['n2']
	n_common = kg_dict['n_common']

	init_ent_emb = None
	init_rel_emb = None

	semi_model = Semi_Rescal(emb_dim=emb_dim, n_ent_list=[n1, n2, n_common], n_rel=kg.n_rel, alpha=ALPHA, device=device,\
							 init_ent_emb=init_ent_emb, init_rel_emb=init_rel_emb, \
							 eps=1e-4, lamda=LAMDA, max_iter=100, thresh=1e-9, w_max_iter=100, w_thresh=1e-9,\
							 stable_sinkhorn=True, data_precision='float', verbose=verbose)

	criterion = MarginLoss(1.0)

	dataloader = DataLoader(kg, batch_size=batch_size, shuffle=True)

	negative_sampler = Negative_Sampler(kg, n1=semi_model.n1, n_common=semi_model.n_common, n2=semi_model.n2)

	optimizer = torch.optim.Adam(semi_model.parameters(), lr=learning_rate, weight_decay=1e-5)

	epochs_iter = tqdm(range(EPOCHS), unit='epoch')
	for epoch in epochs_iter:
		
		total_batch = 0
		running_loss = 0.0

		for h, t, r in dataloader:
			total_batch += 1

			h, t, r = h.to(device), t.to(device), r.to(device)

			n_h, n_t = negative_sampler.corrupt_batch(h, t, r)  # TODO: quiz, can it be used in the combined model?????

			optimizer.zero_grad()
			pos, neg, pos_regul, neg_regul, ot_loss = semi_model(h, t, n_h, n_t, r)

			loss = criterion(pos, neg) + (pos_regul + neg_regul)/2 + ot_loss*ALPHA

			loss.backward()
			optimizer.step()

			running_loss += loss.item()

		ot_cost = semi_model.update_P()  # if alpha = 0.0 => do nothing, reurn torch.tensor(0.0)

		epochs_iter.set_description('Ep: %s | mean loss: %.5f ' %(epoch, running_loss/total_batch))


	# save the emeddings for warmstarting
	ent_embedding = semi_model.ent_emb.weight.detach().to('cpu')
	rel_embedding = semi_model.rel_mat.weight.detach().to('cpu')

	#save_path = 'hyper-search_warmstart_embeddings/'
	save_path = 'warmstart_embeddings/'
	torch.save(ent_embedding, save_path + '%s_WARM-START_overlap=%s_ent_embedding_Emb-dim=%s_Batch-size=%s_lr=%s_EPOCHS=%s_repeat=%s.pt' %(data_name, overlap, emb_dim, batch_size, learning_rate, EPOCHS, repeat))
	torch.save(rel_embedding, save_path + '%s_WARM-START_overlap=%s_rel_embedding_Emb-dim=%s_Batch-size=%s_lr=%s_EPOCHS=%s_repeat=%s.pt' %(data_name, overlap, emb_dim, batch_size, learning_rate, EPOCHS, repeat))
	print('Training for %s epochs! \nSaved embeddings of ent and rel into %s' %(EPOCHS, save_path))


if __name__ == '__main__':
	with torch.cuda.device(0):
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

		for data_name in ['FB15k-237', 'WN18RR', 'DBbook2014', 'ML1M']:
			for data_file, overlap in data_file_dict[data_name]:
				for emb_dim in [100]:
					for repeat in range(10):
						batch_size = 500
						learning_rate = 0.005

						main(data_file, data_name, emb_dim, batch_size, learning_rate, overlap, repeat)





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


MARGIN = 1
LAMDA = 1.0

data_name = 'ML1M'
path = '../data/KG_datasets/%s/semi/divided/fix_intra-test/' %data_name.lower()

overlap = 1.5  # percentage of entity overlapping

# load best hypers
best_hypers_file = 'best_hypers/%s_overlap=%s_best_hypers_MMD.pkl' %(data_name, overlap)
print(best_hypers_file)
with open(best_hypers_file, 'rb') as file:
	best_hypers = pickle.load(file).params

ALPHA = best_hypers['alpha'] 
EMB_DIM = best_hypers['emb_dim'] # fixed to be 100
EPOCHS = best_hypers['EPOCHS'] # fixed to 300
BATCH_SIZE = best_hypers['batch_size']
LEARNING_RATE = best_hypers['lr']


def main(al, inter_prop, repeat_index):

	ALPHA = al

	inter_data_prop = inter_prop

	print('inter data prop: ', inter_data_prop)
	print('alpha: ', ALPHA)
	print('lamda: ', LAMDA)
	print('batch_size: ', BATCH_SIZE)
	print('Epochs: ', EPOCHS)
	print('LEARNING_RATE: ', LEARNING_RATE)
	print('Emb_dim: ', EMB_DIM)

	verbose = False #True #False

	device = torch.device('cuda:%s' %torch.cuda.current_device() if torch.cuda.is_available() else 'cpu')
	print('device: ', device)

	# Load data
	with open(path + 'kg_valid_dict.pkl', 'rb') as file:
		kg_dict = pickle.load(file)

	kg = kg_dict['intra_train']
	intra_kg_test = kg_dict['intra_test']
	inter_kg_test = kg_dict['inter_test']

	inter_kg_valid = kg_dict['inter_valid']

	n1 = kg_dict['n1']
	n2 = kg_dict['n2']
	n_common = kg_dict['n_common']

	print('n1: %s - n_common: %s - n2: %s' %(n1, n_common, n2))
	print('entities overlapped level: %.4f' %(n_common/(n1+n_common)))
	print('Intra train triplets: ', kg.n_facts)
	print('Intra test triplets: ', intra_kg_test.n_facts)
	print('Inter test triplets: ', inter_kg_test.n_facts)
	print('Inter valid triplets: ', inter_kg_valid.n_facts)


	# Load embedding learned with alpha=0. WARM START
	ent_file_name = 'warmstart_embeddings/' + '%s_WARM-START_overlap=%s_ent_embedding_Emb-dim=100_Batch-size=500_lr=0.005_EPOCHS=100_repeat=%s.pt' %(data_name, overlap, repeat_index)
	rel_file_name = 'warmstart_embeddings/' + '%s_WARM-START_overlap=%s_rel_embedding_Emb-dim=100_Batch-size=500_lr=0.005_EPOCHS=100_repeat=%s.pt' %(data_name, overlap, repeat_index)
	init_ent_emb = load_embedding(ent_file_name) #if ALPHA != 0. else None
	init_rel_emb = load_embedding(rel_file_name) #if ALPHA != 0. else None

	# load best hyperparameters
	

	# MODEL
	semi_model = Semi_Rescal(emb_dim=EMB_DIM, n_ent_list=[n1, n2, n_common], n_rel=kg.n_rel, alpha=ALPHA, device=device,\
							 init_ent_emb=init_ent_emb, init_rel_emb=init_rel_emb, \
							 use_MMD=True, \
							 eps=1e-4, lamda=LAMDA, max_iter=100, thresh=1e-9, w_max_iter=100, w_thresh=1e-9,\
							 stable_sinkhorn=True, data_precision='float', verbose=verbose)

	# criterion
	criterion = MarginLoss(MARGIN)

	# dataloader + negative sampler
	dataloader = DataLoader(kg, batch_size=BATCH_SIZE, shuffle=True)


	# check if there is leakage of data
	mix_count = 0
	for head, tail, _ in dataloader:
		for i in range(len(head)):
			if (head[i].item() < n1 + n_common) and (tail[i].item() < n1 + n_common):
				continue
			elif (head[i].item() >= n1) and (tail[i].item() >= n1):
				continue
			else:
				print('mixed triplets: ', head[i].item(), tail[i].item())
				mix_count +=1

	print('Intra mix count: ', mix_count, kg.n_facts)

	mix_count = 0
	for head, tail, _ in DataLoader(inter_kg_test, batch_size=BATCH_SIZE, shuffle=True):
		for i in range(len(head)):
			if (head[i].item() < n1) and (tail[i].item() >= n1 + n_common):
				continue
			elif (head[i].item() >= n1 + n_common) and (tail[i].item() < n1):
				continue
			else:
				print('mixed triplets: ', head[i].item(), tail[i].item())
				mix_count +=1

	print('Inter mix count: ', mix_count, inter_kg_test.n_facts)


	negative_sampler = Negative_Sampler(kg, n1=semi_model.n1, n_common=semi_model.n_common, n2=semi_model.n2)
	#negative_sampler = Negative_Sampler(kg)

	# optimizer
	optimizer = torch.optim.Adam(semi_model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

	# ot_loss_list to save ot_loss after each update
	batch_ot_loss_list = []
	epoch_aveg_loss_list = []
	epoch_ot_loss_list = []


	best_valid_inter_hit10 = 0.0
	best_valid_intra_hit10 = 0.0
	best_valid_epoch = 0

	PATIENCE = 50

	check_gap = 5
	early_stop = 0

	epochs_iter = tqdm(range(EPOCHS), unit='epoch')
	for epoch in epochs_iter:
		t0 = time()

		running_loss = 0.0
		total_batch = 0

		for h, t, r in dataloader:
			tb = time()
			total_batch += 1

			h, t, r = h.to(device), t.to(device), r.to(device)

			n_h, n_t = negative_sampler.corrupt_batch(h, t, r)  # TODO: quiz, can it be used in the combined model?????

			optimizer.zero_grad()
			pos, neg, pos_regul, neg_regul, ot_loss = semi_model(h, t, n_h, n_t, r)

			loss = criterion(pos, neg) + (pos_regul + neg_regul)/2 + ot_loss*ALPHA

			loss.backward()
			optimizer.step()

			batch_ot_loss_list.append(ot_loss.to('cpu').item())

			running_loss += loss.item()


		if semi_model.alpha != 0:
			sinkhorn_cost = semi_model.update_P()
		else:
			sinkhorn_cost = torch.tensor(0)

		epoch_ot_loss_list.append(sinkhorn_cost.to('cpu').item())
		epoch_aveg_loss_list.append((running_loss/total_batch))

		epochs_iter.set_description(
			'Ep %s | mean loss: %.5f | MMD_loss: %.5f | 1 epoch: %.2f s| '
			%(epoch + 1, running_loss/ total_batch, sinkhorn_cost, time() - t0)
		)

		if epoch % check_gap == 0: # 50
			print('########################### epoch: %s ########################' % epoch)
			intra_hit10, _, _, inter_hit10, _, _ = save(batch_ot_loss_list, epoch_ot_loss_list, epoch_aveg_loss_list, data_name, ALPHA, LAMDA, inter_data_prop, semi_model, intra_kg_test, inter_kg_test, inter_kg_valid, repeat_index)
			
			if inter_hit10 > best_valid_inter_hit10:
				early_stop = 0

				best_valid_inter_hit10 = inter_hit10
				best_valid_intra_hit10 = intra_hit10
				best_valid_epoch = epoch

				print('Best valid inter hit@10: %.4f - intra hit@10: %.4f at epoch %s' %(best_valid_inter_hit10, best_valid_intra_hit10, best_valid_epoch))

				# save embedding of best validation
				save_embedding(path='fix-intra_semi_embeddings/', ent_embedding=semi_model.ent_emb, rel_embedding=semi_model.rel_mat, n1=semi_model.n1, n_common=semi_model.n_common, data_name=data_name, ALPHA=ALPHA, LAMDA=LAMDA, inter_data_prop=inter_data_prop, repeat_index=repeat_index)

			else:
				early_stop += check_gap

			if early_stop >= PATIENCE:
				print('=========================== Finished training with early stopping ======================')
				print('Best valid inter hit@10: %.4f - intra hit@10: %.4f at epoch %s' %(best_valid_inter_hit10, best_valid_intra_hit10, best_valid_epoch))
				return save(batch_ot_loss_list, epoch_ot_loss_list, epoch_aveg_loss_list, data_name, ALPHA, LAMDA, inter_data_prop, semi_model, intra_kg_test, inter_kg_test, inter_kg_valid, repeat_index, savefig=True, test=True)

	print('*********************** Finished training *************************************')
	print('Best valid inter hit@10: %.4f - intra hit@10: %.4f at epoch %s' %(best_valid_inter_hit10, best_valid_intra_hit10, best_valid_epoch))
	return save(batch_ot_loss_list, epoch_ot_loss_list, epoch_aveg_loss_list, data_name, ALPHA, LAMDA, inter_data_prop, semi_model, intra_kg_test, inter_kg_test, inter_kg_valid, repeat_index, savefig=True, test=True)


def save(batch_ot_loss_list, epoch_ot_loss_list, epoch_aveg_loss_list, data_name, ALPHA, LAMDA, inter_data_prop, semi_model, intra_kg_test, inter_kg_test, inter_kg_valid, repeat_index, savefig=False, test=False):
	
	if savefig:
		# save plots of ot loss
		plt.plot(batch_ot_loss_list)
		plt.xlabel('batches')
		plt.ylabel('ot loss/batch')
		plt.savefig('fix-intra_semi_plots/%s_MMD_WARM-START_SEMI_batch_ot_loss_alpha=%s_lamda=%s_semi=%s.png' %(data_name, ALPHA, LAMDA, inter_data_prop))
		plt.clf()

		# save plots of ot loss / epochs
		plt.plot(epoch_ot_loss_list)
		plt.xlabel('epochs')
		plt.ylabel('ot loss/epoch')
		plt.savefig('fix-intra_semi_plots/%s_MMD_WARM-START_SEMI_epoch_ot_loss_alpha=%s_lamda=%s_semi=%s.png' %(data_name, ALPHA, LAMDA, inter_data_prop))
		plt.clf()

		# save plots of aveg loss / epochs
		plt.plot(epoch_aveg_loss_list)
		plt.xlabel('epochs')
		plt.ylabel('running aveg loss')
		plt.savefig('fix-intra_semi_plots/%s_MMD_WARM-START_SEMI_running_loss_alpha=%s_lamda=%s_semi=%s.png' %(data_name, ALPHA, LAMDA, inter_data_prop))
		plt.clf()

		with open('fix-intra_semi_plots/%s_MMD_WARM-START_SEMI_every-loss_alpha=%s_lamda=%s_semi=%s.pkl' %(data_name, ALPHA, LAMDA, inter_data_prop), 'wb') as file:
			loss_dict = {
				'batch_ot_loss': batch_ot_loss_list,
				'epoch_ot_loss': epoch_ot_loss_list,
				'running_loss': epoch_aveg_loss_list,
			}
			pickle.dump(loss_dict, file)

	
	if test:
		#Load embeddings of best validation
		semi_model.ent_emb.weight.data = load_embedding('fix-intra_semi_embeddings/' + '%s_MMD_WARM-START_SEMI_all_ent_embedding_alpha=%s_lamda=%s_semi=%s_repeat=%s.pt' %(data_name, ALPHA, LAMDA, inter_data_prop, repeat_index))
		semi_model.rel_mat.weight.data = load_embedding('fix-intra_semi_embeddings/' + '%s_MMD_WARM-START_SEMI_all_rel_embedding_alpha=%s_lamda=%s_semi=%s_repeat=%s.pt' %(data_name, ALPHA, LAMDA, inter_data_prop, repeat_index))


		print('------------- Intra-domain Testing --------------')
		semi_model.is_intra = True  # THIS ONE IS IMPORTANT, must not forget to set semi_model.is_intra = True or False
		evaluator_intra = LinkPredictionEvaluator(semi_model, intra_kg_test)
		evaluator_intra.evaluate(200, 10)
		evaluator_intra.print_results(k=[10])

		print('----------- Inter-domain Testing --------------')
		semi_model.is_intra = False
		evaluator_inter = LinkPredictionEvaluator(semi_model, inter_kg_test)
		evaluator_inter.evaluate(200, 10)
		evaluator_inter.print_results(k=[10])

	else: # validation
		print('------------- Intra-domain Testing --------------')
		semi_model.is_intra = True  # THIS ONE IS IMPORTANT, must not forget to set semi_model.is_intra = True or False
		evaluator_intra = LinkPredictionEvaluator(semi_model, intra_kg_test)
		evaluator_intra.evaluate(200, 10)
		evaluator_intra.print_results(k=[10])

		print('----------- Inter-domain Testing --------------')
		semi_model.is_intra = False
		evaluator_inter = LinkPredictionEvaluator(semi_model, inter_kg_valid)
		evaluator_inter.evaluate(200, 10)
		evaluator_inter.print_results(k=[10])


	return evaluator_intra.hit_at_k(k=10)[1], evaluator_intra.mean_rank()[1], evaluator_intra.mrr()[1], \
			evaluator_inter.hit_at_k(k=10)[1], evaluator_inter.mean_rank()[1], evaluator_inter.mrr()[1]


def save_embedding(path, ent_embedding, rel_embedding, n1, n_common, data_name, ALPHA, LAMDA, inter_data_prop, repeat_index):
	ent_embedding = ent_embedding.weight.detach().to('cpu')
	rel_embedding = rel_embedding.weight.detach().to('cpu')
	#torch.save(ent_embedding[:n1 + n_common], path + '%s_MMD_WARM-START_SEMI_embedding_1_alpha=%s_lamda=%s_semi=%s_repeat=%s.pt' %(data_name, ALPHA, LAMDA, inter_data_prop, repeat_index))
	#torch.save(ent_embedding[n1:], path + '%s_MMD_WARM-START_SEMI_embedding_2_alpha=%s_lamda=%s_semi=%s_repeat=%s.pt' %(data_name, ALPHA, LAMDA, inter_data_prop, repeat_index))
	
	torch.save(ent_embedding, path + '%s_MMD_WARM-START_SEMI_all_ent_embedding_alpha=%s_lamda=%s_semi=%s_repeat=%s.pt' %(data_name, ALPHA, LAMDA, inter_data_prop, repeat_index))
	torch.save(rel_embedding, path + '%s_MMD_WARM-START_SEMI_all_rel_embedding_alpha=%s_lamda=%s_semi=%s_repeat=%s.pt' %(data_name, ALPHA, LAMDA, inter_data_prop, repeat_index))


def load_embedding(filename):
	
	try:
		emb = torch.load(filename)
		print('############################ MMD_Warm-start! Loaded embedidng from %s #######################' % filename)
		return emb
	except:
		print('############################ Do not have Embeddings learned with ALPHA=0., initialize embeddings with random ############################')
		return None
	

repeat = 10
# context to set default torch.cuda.device()
with torch.cuda.device(1):
	for inter_prop in [0.0]: #[0.1, 0.3]:
		#for al in [5.0]: #[0., 0.5, 1.5, 3, 5.]:

		al = ALPHA

		intra_hit_list, intra_mean_list, intra_mrr_list = [], [], []
		inter_hit_list, inter_mean_list, inter_mrr_list = [], [], []

		for idx in range(repeat):

			intra_hit, intra_mean, intra_mrr, inter_hit, inter_mean, inter_mrr = main(al, inter_prop, idx)

			intra_hit_list.append(intra_hit)
			intra_mean_list.append(intra_mean)
			intra_mrr_list.append(intra_mrr)

			inter_hit_list.append(inter_hit)
			inter_mean_list.append(inter_mean)
			inter_mrr_list.append(inter_mrr)

		with open('fix-intra_semi_results/%s_MMD_WARM-START_SEMI_AVERAGE_alpha=%s_lamda=%s_semi=%s.txt' % (data_name, al, LAMDA, inter_prop), 'w') as file:
			file.write('\nSEMI Average results of %s\n\n' % (data_name))

			file.write('Intra Testing (Filtered)\n')
			file.write('Hit@10\tMean Rank\tMRR\n')
			file.write('%.3f\t%d\t%.3f\n' % (torch.mean(torch.tensor(intra_hit_list)), torch.mean(torch.tensor(intra_mean_list)), torch.mean(torch.tensor(intra_mrr_list))))
			file.write('%.3f\t%d\t%.3f\n' % (torch.std(torch.tensor(intra_hit_list)), torch.std(torch.tensor(intra_mean_list)), torch.std(torch.tensor(intra_mrr_list))))

			file.write('\n\n')

			file.write('Inter Testing (Filtered)\n')
			file.write('Hit@10\tMean Rank\tMRR\n')
			file.write('%.3f\t%d\t%.3f\n' % (torch.mean(torch.tensor(inter_hit_list)), torch.mean(torch.tensor(inter_mean_list)), torch.mean(torch.tensor(inter_mrr_list))))
			file.write('%.3f\t%d\t%.3f\n' % (torch.std(torch.tensor(inter_hit_list)), torch.std(torch.tensor(inter_mean_list)), torch.std(torch.tensor(inter_mrr_list))))

torch.cuda.empty_cache()


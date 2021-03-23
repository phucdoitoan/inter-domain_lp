

import torch
import torch.nn as nn

#from torchkge.models import RESCALModel as Rescal 
from utils import Regularized_Rescal as Rescal
#from rescal_autograd import Rescal

from gw_ot.wgw import Entropic_WGW
from torchkge.utils import get_true_targets, get_rank

from mmd.mmd import MMD_loss


class Semi_Rescal(Rescal):

	def __init__(self, emb_dim, n_ent_list, n_rel, alpha, device, init_ent_emb, init_rel_emb, use_MMD=False, **kwargs):

		#super(Semi_Rescal, self).__init__(ent_num=sum(n_ent_list), rel_num=n_rel, rank=emb_dim)
		super(Semi_Rescal, self).__init__(emb_dim=emb_dim, n_entities=sum(n_ent_list), n_relations=n_rel)

		if init_ent_emb is not None:
			self.ent_emb.weight.data = init_ent_emb
		if init_rel_emb is not None:
			self.rel_mat.weight.data = init_rel_emb

		if device != 'cpu':
			super().to(device)

		#self.n_model = len(n_ent_list)
		#self.n_ent_list = n_ent_list

		self.n1 = n_ent_list[0]
		self.n2 = n_ent_list[1]

		if len(n_ent_list) == 2:
			self.n_common = 0
		else:
			self.n_common = n_ent_list[2]

		self.n_rel = n_rel

		self.alpha = alpha

		self.device = device

		self.use_MMD = use_MMD

		if not self.use_MMD:
			self.P = torch.ones((self.n1+self.n_common, self.n2 + self.n_common), dtype=torch.float, device=device)
			self.P /= self.P.sum()
			#print('self.P: ', self.P.shape)

			self.wgw = Entropic_WGW(**kwargs)
		else:
			self.mmd = MMD_loss()


	def forward(self, pos_h, pos_r, neg_h, neg_t, rel):

		pos, neg, pos_regul, neg_regul = super().forward(pos_h, pos_r, neg_h, neg_t, rel)

		#print('pos_regul: ', pos_regul)
		#print('neg_regul: ', neg_regul)
		#pos, neg, *_ = super().forward(pos_h, pos_r, neg_h, neg_t, rel)# using the torchkge.models.RESCALModel withou regularization
		#pos_regul, neg_regul = 0.0, 0.0

		ot_loss = torch.tensor(0.0)

		if self.alpha != 0:
			ent_all = torch.cat([pos_h, pos_r, neg_h, neg_t])
			ent_1 = self.filter_ent(ent_all, 0, self.n1+self.n_common)
			ent_2 = self.filter_ent(ent_all, self.n1, self.n1+self.n_common+self.n2)
			#ent_2 -= n1
			if not self.use_MMD:
				ot_loss = self.wgw_loss(ent_1, ent_2)
				#ot_loss *= self.alpha
			else:
				emb1 = self.ent_emb.weight[ent_1]
				emb2 = self.ent_emb.weight[ent_2]

				ot_loss = self.mmd(emb1, emb2, n1=self.n1+self.n_common, n2=self.n2+self.n_common)

		return pos, neg, pos_regul, neg_regul, ot_loss

	def wgw_loss(self, ent_1, ent_2):
		
		emb1 = self.ent_emb.weight[ent_1]
		emb2 = self.ent_emb.weight[ent_2]

		scaled_ent_2 = ent_2 - self.n1
		P_sliced = self.P[ent_1][:, scaled_ent_2] 

		# compute w_cost
		if self.wgw.lamda == 0.:
			w_cost = torch.tensor(0.)
		else:
			norm = self.wgw.cost_matrix(emb1, emb2, cost_type='L2')
			w_cost = torch.sum(norm * P_sliced)

		# compute gw_cost
		if self.wgw.lamda == 1.:
			gw_cost = torch.tensor(0.)
		else:
			C1 = self.wgw.cost_matrix(emb1, emb1, cost_type=self.wgw.intra_cost_type)
			C2 = self.wgw.cost_matrix(emb2, emb2, cost_type=self.wgw.intra_cost_type)

			L = self.wgw.tensor_matrix_mul(C1, C2, P_sliced)

			gw_cost = (P_sliced * L).sum()

		if self.wgw.verbose:
			print('\tw_cost: ', w_cost)
			print('\tgw_cost: ', gw_cost)

		return self.wgw.lamda*w_cost + (1-self.wgw.lamda)*gw_cost

	def update_P(self):
		# update transport plan P after some epochs of training
		# n1: 0 ~ (n1-1) entities belong to the first domain
		# n2: n2 ~ : entities belong to the second domain

		if self.alpha == 0.0:
			return torch.tensor(0.0)

		if not self.use_MMD:
			# compute OT loss and update the OT plan P
			with torch.no_grad():
				skn_cost, tmp_P = self.wgw(self.ent_emb.weight[:self.n1+self.n_common], self.ent_emb.weight[self.n1:])
				self.P = tmp_P

				if self.wgw.verbose:
					print('tmp_P: ', tmp_P.sum(), '\n', tmp_P)
					print('tmp_P: ', tmp_P.sum(dim=0), tmp_P.sum(dim=1))

			return skn_cost

		else:
			# compute the mmd loss
			with torch.no_grad():
				mmd_cost = self.mmd(self.ent_emb.weight[:self.n1+self.n_common], self.ent_emb.weight[self.n1:])

			return mmd_cost


	def filter_ent(self, l, thresh1, thresh2):
		mask1 = l >= thresh1
		mask2 = l < thresh2
		mask = mask1 * mask2

		return l[mask]

	#reference: line 168 - 234 in https://github.com/torchkge-team/torchkge/blob/master/torchkge/models/interfaces.py
	def lp_compute_ranks(self, e_emb, candidates, r, e_idx, r_idx, true_idx, dictionary, heads=1):
		b_size = r_idx.shape[0]
		
		if heads == 1:
			scores = self.lp_scoring_function(e_emb, candidates, r)
		else:
			scores = self.lp_scoring_function(candidates, e_emb, r)

		# filter out the true negative samples by assigning -inf score
		filt_scores = scores.clone()
		for i in range(b_size):
			true_targets = get_true_targets(dictionary, e_idx, r_idx, true_idx, i)

			if true_targets is None:
				continue
			filt_scores[i][true_targets] = - float('Inf')

		origin_scores = scores.clone()
		origin_filt_scores = filt_scores.clone()

		# filter entities out of range when evalute intra-domain or inter-domain

		if self.is_intra:  # evaluating intra-domain prediction
			#print('Evaluating Intra-domain....')
			
			for i in range(len(e_idx)): 
				if e_idx[i] < self.n1:  # ent in domain 1
					scores[i][self.n1+self.n_common:] = - float('Inf')  # set scores of ent in domain 2 to -Inf
					filt_scores[i][self.n1+self.n_common:] = - float('Inf')

				elif e_idx[i] >= self.n1 + self.n_common:  # ent in domain 2
					scores[i][:self.n1] = - float('Inf')  # set scores of ent in domain 1 to -Inf
					filt_scores[i][:self.n1] = - float('Inf')

				else:  # ent is the common ent
					if true_idx[i] < self.n1:  # true ent in domain 1
						scores[i][self.n1+self.n_common:] = - float('Inf')  # set scores of ent in domain 2 to -Inf
						filt_scores[i][self.n1+self.n_common:] = - float('Inf')
					elif true_idx[i] >= self.n1 + self.n_common:
						scores[i][:self.n1] = - float('Inf')  # set scores of ent in domain 1 to -Inf
						filt_scores[i][:self.n1] = - float('Inf')
					else:
						if torch.rand(1) > 0.5:
							scores[i][self.n1+self.n_common:] = - float('Inf')  # set scores of ent in domain 2 to -Inf
							filt_scores[i][self.n1+self.n_common:] = - float('Inf')
						else:
							scores[i][:self.n1] = - float('Inf')  # set scores of ent in domain 1 to -Inf
							filt_scores[i][:self.n1] = - float('Inf')

		else:  # evaluating inter-domain prediction
			#print('Evaluating Inter-domain.....')

			for i in range(len(e_idx)): 
				if e_idx[i] < self.n1:  # ent in domain 1
					scores[i][:self.n1+self.n_common] = - float('Inf')  # set scores of ent in domain 1 to -Inf
					filt_scores[i][:self.n1+self.n_common] = - float('Inf')

				elif e_idx[i] >= self.n1 + self.n_common: # e_idx[i] >= self.n1 + n_common:  # ent in domain 2
					scores[i][self.n1:] = - float('Inf')  # set scores of ent in domain 2 to -Inf
					filt_scores[i][self.n1:] = - float('Inf') 

				else:
					print('WARNING! NO GOOD, common entities appear in inter-domain test triplets')

		# from dissimilarities, extract the rank of the true entity
		rank_true_entities = get_rank(scores, true_idx)
		filtered_rank_true_entities = get_rank(filt_scores, true_idx)

		#origin_rank_true_entities = get_rank(origin_scores, true_idx)

		#print('e_idx: ', e_idx.tolist())
		#print('ent_d1_idx: ', ent_d1_idx.tolist())
		#print('ent_d2_idx: ', ent_d2_idx.tolist())

		#print('scores: ', scores.shape)

		#print('true_idx: ', true_idx.tolist())

		#print('rank_true_entities: ', rank_true_entities.tolist())
		#print('origin rank_true_entities: ', origin_rank_true_entities.tolist())
		#print('scores true entitites: ', scores.min().item())
		#print('ahahahaha: ', filt_scores.min().item())

		return rank_true_entities, filtered_rank_true_entities













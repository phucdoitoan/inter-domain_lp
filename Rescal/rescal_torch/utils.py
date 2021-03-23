


import torch
from torchkge.data_structures import KnowledgeGraph
from torchkge.exceptions import SanityError
from collections import defaultdict
from torch import Tensor as Tensor
from torchkge.models import RESCALModel as Rescal

from torchkge.evaluation import LinkPredictionEvaluator

import time


# helper function
def _get_true_targets(dictionary, e_idx, r_idx, true_idx, i):
	# refer to : https://github.com/torchkge-team/torchkge/blob/master/torchkge/utils/modeling.py
	"""

	Args:
		dictionary: keys (ent_idx, rel_idx), values: list of ent_idx s.t the triplet is true fact
		e_idx: torch.Tensor, shape: (batch_size) Long tensor of ent_idx
		r_idx: shape: (batch_size), Long Tensor of rel_idx
		true_idx: Long Tensor of ent_idx, s.t (e_idx, r_idx, true_idx) is a true fact
		i: the index of the batch is currently treated

	Returns:
		true_targets: Long Tensor of ent_idx s.t true_targets != true_idx and (e_idx, r_idx, and ent in true_targets) is a true fact
	"""

	true_targets = dictionary[e_idx[i].item(), r_idx[i].item()].copy()

	if len(true_targets) == 1:
		return None
	true_targets.remove(true_idx[i].item())

	return torch.LongTensor(list(true_targets))

def _get_rank(scores, true, low_values=False):
	# refer to : https://github.com/torchkge-team/torchkge/blob/master/torchkge/utils/operations.py
	"""
	Compute the rank of entity at index true[i]
	Args:
		scores: torch.Tensor, shape: (b_size, n_ent), scores for each entities
		true: (b_size,)
		true[i] is the index of the true entity in the batch
		low_values: If True, best rank is the lowest score, else it is the highest

	Returns:
		ranks: torch.Tensor, shape: (b_size) ranks of the true entities in the batch
		ranks[i] - 1 is the number of entities which have better scores in scores than the one with index true[i]
	"""
	true_scores = scores.gather(dim=1, index=true.long().view(-1, 1))

	if low_values:
		return (scores < true_scores).sum(dim=1) + 1
	else:
		return (scores > true_scores).sum(dim=1) + 1


class UniformNegativeSampler:
	def __init__(self, kg, n_neg=1, **kwargs):
		self.kg = kg
		self.n_ent = kg.n_ent
		self.n_facts = kg.n_facts

		self.n_neg = n_neg

	def corrupt_batch(self, heads, tails, relations=None, n_neg=None, low=None, high=None):
		"""Sample negative examples from positive examples, according to Bordes et al. 2013"""
		if n_neg is None:
			n_neg = self.n_neg

		device = heads.device
		assert (device == tails.device), 'heads and tails must be on a same device'

		batch_size = heads.shape[0]
		neg_heads = heads.repeat(n_neg)
		neg_tails = tails.repeat(n_neg)

		# Randomly choose which samples will have head/tail corrupted
		mask = torch.bernoulli(torch.ones(batch_size * n_neg, device=device)/2).double()

		n_h_cor = int(mask.sum().item())

		if low is None:
			low = 0
		if high is None:
			high = self.n_ent

		neg_heads[mask==1] = torch.randint(low=low, high=high, size=(n_h_cor,), device=device)
		neg_tails[mask==0] = torch.randint(low=low, high=high, size=(batch_size * n_neg - n_h_cor,), device=device)

		return neg_heads.long(), neg_tails.long()


class UniformNegativeSampler_Extended(UniformNegativeSampler):
	def __init__(self, kg, n_neg=1, n1=None, n_common=None, n2=None):
		super(UniformNegativeSampler_Extended, self).__init__(kg, n_neg)

		self.n1 = n1
		self.n_common = n_common
		self.n2 = n2

		assert self.n1 + self.n_common + self.n2 == self.n_ent, 'The sum of n1, n_common, and n2 must be equal to n_ent of kg: %s vs %s' \
																% (self.n1 + self.n_common + self.n2, self.n_ent)

	def corrupt_batch(self, heads, tails, relations=None, n_neg=None):
		#print('************************* E1: %s ~ %s --- E2: %s ~ %s ********************** ' % (0, self.n1+self.n_common, self.n1, self.n1+self.n_common+self.n2))
		#print('heads: ', heads)
		#print('tails: ', tails)


		if n_neg is None:
			n_neg = self.n_neg

		neg_heads = heads.repeat(n_neg)
		neg_tails = tails.repeat(n_neg)

		mask1 = self.filter(heads, tails, 0, self.n1 + self.n_common)
		mask2 = self.filter(heads, tails, self.n1, self.n1 + self.n_common + self.n2)

		heads1, tails1 = heads[mask1], tails[mask1]
		heads2, tails2 = heads[mask2], tails[mask2]

		#print('heads1: ', heads1)
		#print('tails1: ', tails1)
		#print('heads2: ', heads2)
		#print('tails2: ', tails2)


		neg_h1, neg_t1 = super().corrupt_batch(heads1, tails1, low=0, high=self.n1 + self.n_common)
		neg_h2, neg_t2 = super().corrupt_batch(heads2, tails2, low=self.n1, high=self.n1 + self.n_common + self.n2)

		if torch.rand(1) > 0.5:
			neg_heads[mask1] = neg_h1
			neg_tails[mask1] = neg_t1
			neg_heads[mask2] = neg_h2
			neg_tails[mask2] = neg_t2
		else:
			neg_heads[mask2] = neg_h2
			neg_tails[mask2] = neg_t2
			neg_heads[mask1] = neg_h1
			neg_tails[mask1] = neg_t1

		#print('neg_heads: ', neg_heads.long())
		#print('neg_tails: ', neg_tails.long())

		return neg_heads.long(), neg_tails.long()


	def filter(self, heads, tails, thresh1, thresh2):
 		mask1 = (heads >= thresh1) * (tails >= thresh1)
 		mask2 = (heads < thresh2) * (tails < thresh2)
 		mask = mask1 * mask2

 		return mask


class Extended_KnowledgeGraph(KnowledgeGraph):

	def __init__(self, kg=None, **kwargs):
		"""
		kg: torchkge KnowledgeGraph
		"""

		if kg is None:
			super(Extended_KnowledgeGraph, self).__init__(**kwargs)
		else:
			super(Extended_KnowledgeGraph, self).__init__(kg= {'heads': kg.head_idx,
				                                               'tails': kg.tail_idx,
				                                               'relations': kg.relations},\
														  ent2ix=kg.ent2ix,\
														  rel2ix=kg.rel2ix,\
														  dict_of_heads=kg.dict_of_heads,\
														  dict_of_tails=kg.dict_of_tails)

	def extend_kg(self, df=None, kg=None):
		"""
		df: pd DataFrame, columns' names are 'from', 'to', 'rel'
		"""

		if df is not None:
			# update self.ent2ix and self.rel2ix:
			df_ent = set(df['from'].to_list()).union(set(df['to'].to_list()))
			df_rel = set(df['rel'].to_list())

			new_ent = df_ent.difference(set(self.ent2ix.keys()))
			new_rel = df_rel.difference(set(self.rel2ix.keys()))

			for e in new_ent:
				e_ix = len(self.ent2ix)
				self.ent2ix[e] = e_ix

			for r in new_rel:
				r_ix = len(self.rel2ix)
				self.rel2ix[r] = r_ix

			# update n_facts, n_ent, n_rel
			self.n_facts += len(df)
			self.n_ent = len(self.ent2ix)
			self.n_rel = len(self.rel2ix)

			# update head_idx, tail_idx, relations
			new_head_idx = torch.tensor(df['from'].map(self.ent2ix).values).long()
			new_tail_idx = torch.tensor(df['to'].map(self.ent2ix).values).long()
			new_relations = torch.tensor(df['rel'].map(self.rel2ix).values).long()

			self.head_idx = torch.cat([self.head_idx, new_head_idx], dim=0)
			self.tail_idx = torch.cat([self.tail_idx, new_tail_idx], dim=0)
			self.relations = torch.cat([self.relations, new_relations], dim=0)

			# update self.dict_of_heads, dict_of_tails
			for h_idx, t_idx, r_idx in zip(new_head_idx, new_tail_idx, new_relations):
				self.dict_of_heads[(t_idx.item(), r_idx.item())].add(h_idx.item())
				self.dict_of_tails[(h_idx.item(), r_idx.item())].add(t_idx.item())

		else: # df is None => kg is not None
			pass

		try:
			self.sanity_check()
		except AssertionError:
			raise SanityError("Please check the sanity of arguments.")

	def split_kg(self, **kwargs):

		splited_kgs = super().split_kg(**kwargs)

		return (Extended_KnowledgeGraph(kg=kg) for kg in splited_kgs)



class Regularized_Rescal(Rescal):
	def __init__(self, reg_lamda=1.0, **kwargs):
		super(Regularized_Rescal, self).__init__(**kwargs)

		self.reg_lamda = reg_lamda

	def regularization(self, sub, obj, rel):
		h = self.ent_emb.weight[sub]
		t = self.ent_emb.weight[obj]
		r = self.rel_mat.weight[rel]

		regul = (torch.mean(h**2) + torch.mean(t**2) + torch.mean(r**2)) / 3
		regul *= self.reg_lamda

		return regul

	def forward(self, heads, tails, negative_heads, negative_tails, relations):

		assert heads.shape[0] == negative_heads.shape[0], 'The length of positive and negative examples must be equal'

		pos, neg = super().forward(heads, tails, negative_heads, negative_tails, relations)

		if self.reg_lamda == 0:
			return pos, neg, 0.0, 0.0
		else:
			pos_regul = self.regularization(heads, tails, relations)
			neg_regul = self.regularization(negative_heads, negative_tails, relations)
			return pos, neg, pos_regul, neg_regul


class Extended_LinkPredictionEvaluator(LinkPredictionEvaluator):

	def __init__(self, model, knowledge_graph, is_intra=True):
		super(Extended_LinkPredictionEvaluator, self).__init__(model, knowledge_graph)

		# evaluate inter or intra domain link prediction
		model.is_intra = is_intra  # set is_intra of model to is_intra



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
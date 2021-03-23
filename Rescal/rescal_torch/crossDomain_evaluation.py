


from torch.utils.data import DataLoader, Dataset
from torch import empty
import torch



class cross_LinkPredictionEvaluator(object):

	def __init__(self, model_s, model_t, kg_s, kg_t):

		self.model_s = model_s
		self.model_t = model_t
		self.kg_s = kg_s
		self.kg_t = kg_t

		self.device = model_s.device
		#self.device = 'cpu'

		self.rank_true_entities = torch.tensor([], device=self.device).long()


	def get_rank(self, scores, true, low_values=False):

		#print('In get_rank: scores: %s - true: %s' %(scores.device, true.device))

		true_scores = scores.gather(dim=1, index=true.long().view(-1, 1))

		if low_values:
			return (scores < true_scores).sum(dim=1) + 1
		else:
			return (scores > true_scores).sum(dim=1) + 1


	def lp_scoring_function(self, h_emb, t_emb, r_emb):
		"""
		Given an entities e_s, r_s of entities in the source domain, compute the scores of triplets (e_s, r_s, e_t) or (e_t, r_s, e_s)
		for all e_t in the target domain. 
		Args:
			h_emb: shape (b_size, emb_dim) or (b_size, n_ent_t, emb_dim)
			t_emb: shape (b_size, n_ent_t, emb_dim) or (b_size, emb_dim)
			r_emb: shape (b_size, emb_dim, emb_dim)
		
		Returns:
			scores: shape (b_size, n_ent_t)
		"""

		b_size = h_emb.shape[0]

		if len(h_emb.shape) == 2 and len(t_emb.shape) == 3:
			# (e_s, r_s, e_t) case

			h_emb = h_emb.unsqueeze(1)
			scores = (h_emb @ r_emb) * t_emb

			return scores.sum(dim=-1)

		else:
			# (e_t, r_s, e_s) case

			t_emb = t_emb.unsqueeze(1)
			scores = (h_emb @ r_emb) * t_emb

			return scores.sum(dim=-1)


	def lp_prep_cands(self, h_idx, t_idx, r_idx, source_is_head=True):

		b_size = h_idx.shape[0]

		r_emb = self.model_s.rel_embedding[r_idx]

		if source_is_head:
			h_emb = self.model_s.ent_embedding[h_idx]
			t_emb = self.model_t.ent_embedding[t_idx]
		else:
			h_emb = self.model_t.ent_embedding[h_idx]
			t_emb = self.model_s.ent_embedding[t_idx]

		candidates = self.model_t.ent_embedding.data.unsqueeze(0)
		candidates = candidates.expand(b_size, -1, -1)

		return h_emb, t_emb, candidates, r_emb


	def lp_compute_ranks(self, e_emb, candidates, r_emb, e_idx, r_idx, true_idx, dictionary=None, source_is_head=True):

		b_size = r_idx.shape[0]

		if source_is_head:
			scores = self.lp_scoring_function(e_emb, candidates, r_emb)
		else:
			scores = self.lp_scoring_function(candidates, e_emb, r_emb)


		# there is no filter case here, because there is no training data across domains


		rank_true_entities = self.get_rank(scores, true_idx)

		return rank_true_entities


	def lp_helper(self, h_idx, t_idx, r_idx, source_is_head=True):

		h_emb, t_emb, candidates, r_emb = self.lp_prep_cands(h_idx, t_idx, r_idx, source_is_head)

		if source_is_head:
			rank_true_entities = self.lp_compute_ranks(h_emb, candidates, r_emb, h_idx, r_idx, t_idx, dictionary=None, source_is_head=True)
		else:
			rank_true_entities = self.lp_compute_ranks(t_emb, candidates, r_emb, t_idx, r_idx, h_idx, dictionary=None, source_is_head=False)

		return rank_true_entities



	def evaluate(self, b_size, k_max, df, source_is_head=True):
		"""
		headers of df must be ['head', 'rel', 'tail']
		"""
		
		self.k_max = k_max

		cross_Triplets_dataset = crossDomain_Triplets(self.kg_s, self.kg_t, df, source_is_head)

		dataloader = DataLoader(cross_Triplets_dataset, batch_size=b_size, shuffle=True, )

		rank_true_entities = torch.empty(size=(df.shape[0],), device=self.device).long()

		for i, batch in enumerate(dataloader):
			
			#print('batch device: ', batch[0].device)
			#print('batch[0] : ', batch[0], '\n*****************************************')
			#print('batch[1] : ', batch[1], '\n******************************************')
			#print('batch[2] : ', batch[2], '\n*************************************')

			h_idx, t_idx, r_idx = batch[0].to(self.device), batch[1].to(self.device), batch[2].to(self.device)
			#h_idx, t_idx, r_idx = batch[0], batch[1], batch[2]

			rk_true_ents = self.lp_helper(h_idx, t_idx, r_idx, source_is_head)

			rank_true_entities[i*b_size: (i+1)*b_size] = rk_true_ents

		self.rank_true_entities = torch.cat((self.rank_true_entities, rank_true_entities), dim=0)

		self.evaluated = True


	def mean_rank(self):
		
		if not self.evaluated:
			raise NotimplementedError('Evaluator has not been evaluated yet')

		return self.rank_true_entities.float().mean().item()


	def hit_at_k(self, k=10):
		
		if not self.evaluated:
			raise NotimplementedError('Evaluator has not been evaluated yet')

		hit = (self.rank_true_entities <= k).float().mean()

		return hit.item()


	def mrr(self):
		
		return (self.rank_true_entities.float() ** (-1)).mean().item()


	def print_results(self, k=None, n_digits=4):

		print('Cross Domain Link Prediction')
		
		if k is None:
			k = 10

		if k is not None and type(k) == int:
			print('Hit@{}: {}'.format(k, round(self.hit_at_k(k=k), n_digits)))

		if k is not None and type(k) == list:
			for i in k:
				print('Hit@{}: {}'.format(i, round(self.hit_at_k(k=i), n_digits)))

		print('Mean Rank: {}'.format(int(self.mean_rank())))
		print('MRR: {}'.format(round(self.mrr(), n_digits)))




class crossDomain_Triplets(Dataset):
	
	def __init__(self, kg_s, kg_t, df, source_is_head=True):

		self.source_is_head = source_is_head

		self.rel_idx = torch.LongTensor(df['rel'].map(kg_s.rel2id).values)

		if source_is_head:
			self.head_idx = torch.LongTensor(df['head'].map(kg_s.ent2id).values)
			self.tail_idx = torch.LongTensor(df['tail'].map(kg_t.ent2id).values)
		else:
			self.head_idx = torch.LongTensor(df['head'].map(kg_t.ent2id).values)
			self.tail_idx = torch.LongTensor(df['tail'].map(kg_s.ent2id).values)

		print('LEN of crossDomain Triplets: ', len(self.head_idx), len(self.tail_idx), len(self.rel_idx))
		
		#print('kg_s.ent2id: ', kg_s.ent2id)
		#print('kg_t.ent2id: ', kg_t.ent2id)

		#self._check_positivity()

		#print('len(df): ', len(df))

		"""
		for i in range(len(df)):
			try:
				h = torch.LongTensor(kg_s.ent2id[df['head'][i]])
			except:
				print("i: %s - df['head'][i]: %s" %(i, df['head'][i]))

			try:	
				t = torch.LongTensor(kg_t.ent2id[df['tail'][i]])
			except:
				print("i: %s - df['tail'][i]: %s" %(i, df['tail'][i]))

			r = torch.LongTensor(kg_s.rel2id[df['rel'][i]])
			try:
				if h < 0:
					print('in head: ', h, df['head'][i])
			except:
				pass
				#print(h)
			try:
				if t < 0:
					print('in tail: ', t, df['tail'][i])
			except:
				pass
				#print(t)
		"""


	def __len__(self):
		return self.head_idx.shape[0]


	def __getitem__(self, i):
		return self.head_idx[i], self.tail_idx[i], self.rel_idx[i]


	def _check_positivity(self):

		heads_neg = torch.where(self.head_idx < 0)
		tails_neg = torch.where(self.tail_idx < 0)
		rels_neg = torch.where(self.rel_idx < 0)

		#print('heads_neg: ', heads_neg)
		#print('tails_neg: ', tails_neg)
		#print('rels_neg: ', rels_neg)








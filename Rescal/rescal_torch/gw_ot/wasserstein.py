


import torch
import torch.nn as nn


class Base_Wasserstein(nn.Module):
	"""
	Base class for original Sinkhorn algorithm and stabilized version
	"""
	def __init__(self, eps, max_iter, thresh=1e-5, verbose=False, data_precision='double'):

		super(Base_Wasserstein, self).__init__()

		self.eps = eps
		self.max_iter = max_iter
		self.thresh = thresh

		self.verbose = verbose
		self.data_precision = data_precision

	def forward(self, x, y, p=None, q=None, cost_type='L2'):

		if p is None:
			p = torch.ones(x.shape[0], device=x.device)
			p /= p.sum()
		if q is None:
			q = torch.ones(y.shape[0], device=y.device)
			q /= q.sum()
        
		if self.data_precision == 'double':
			x = x.double()
			y = y.double()
			p = p.double()
			q = q.double()
		else:
			pass

		C = self.cost_matrix(x, y, cost_type)

		return self.forward_with_cost_matrix(C, p, q)

	def forward_with_cost_matrix(self, C, p, q):
		raise NotImplementedError('Classes inherited from Base_Wasserstein needs to implement this method')

	def cost_matrix(self, x, y, cost_type='L2'):

		if cost_type == 'L2':
			x_row = x.unsqueeze(-2)
			y_col = y.unsqueeze(-3)
			C = torch.sum((x_row - y_col)**2, dim=-1)

		else:
			raise NotImplementedError('The cost type %s is not implemented!' %(cost_type))

		return C


class Entropic_Wasserstein(Base_Wasserstein):

	"""
	Original Sinkhorn algorithm for computing the entropic regularized Wasserstein distance
	Faster than the Stabilized_Entropic_Wassertein() but prone to numerical overflow, underflow
	when eps is considerably smaller than elements of C.

	Reference: 
		Computational Optimal Transport, chapter 4.2
	"""

	def __init__(self, eps, max_iter, thresh=1e-5, verbose=False, data_precision='double'):

		super(Entropic_Wasserstein, self).__init__(eps, max_iter, thresh, verbose, data_precision)

	def forward_with_cost_matrix(self, C, p, q):
		
		K = torch.exp(- C / self.eps)
		b = torch.ones_like(q)

		for it in range(self.max_iter):
			b_old = b

			a = p / (K @ b)
			b = q / (K.T @ a)

			err = torch.norm(b - b_old)
			if err < self.thresh:
				if self.verbose:
					print('Break in Sinkhorn alg at %s-th iteration: Err = %f' %(it, err))

				break

		T = torch.diag(a) @ K @ torch.diag(b)
		w_cost = (T * C).sum()

		return w_cost, T


class Stabilized_Entropic_Wasserstein(Base_Wasserstein):

	"""
	Stabilized version of Entropic_Wasserstein(), the updates are computed in log-domain.
	Stable w.r.t the values of eps, i.e does not prone to numerical overflow and undeflow.
	However, (quite) slower than Entropic_Wassertein.

	Reference:
		Computational Optimal Transport, chapter 4.4, equations 4.43, 4.44
	"""

	def __init__(self, eps, max_iter, thresh=1e-5, verbose=False, data_precision='double'):

		super(Stabilized_Entropic_Wasserstein, self).__init__(eps, max_iter, thresh, verbose, data_precision)

	def forward_with_cost_matrix(self, C, p, q):
		
		f = torch.zeros_like(p)
		g = torch.zeros_like(q)

		for it in range(self.max_iter):
			f_old = f

			f = f + self.eps * (torch.log(p) - torch.logsumexp(self.S(C, f, g), dim=-1))
			g = g + self.eps * (torch.log(q) - torch.logsumexp(self.S(C, f, g).T, dim=-1))

			err = torch.norm(f - f_old)
			if err < self.thresh:
				if self.verbose:
					print('Break in Sinkhorn alg at %s-th iteration: Err = %f' %(it, err))

				break

		P = torch.exp(self.S(C, f, g))
		w_cost = (P * C).sum()

		return w_cost, P

	def S(self, C, f, g):
		""" function S as in 4.43, 4.44, for computing in log-domain """
		return (- C + f.unsqueeze(-1) + g.unsqueeze(-2)) / self.eps


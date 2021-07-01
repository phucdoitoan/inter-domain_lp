


import torch
import torch.nn as nn
from .wasserstein import Entropic_Wasserstein, Stabilized_Entropic_Wasserstein

from time import time


class Entropic_WGW(nn.Module):

	"""
	Computed the entropic regularized gromov-wassertsein discrepancy

	Reference:
		Computational Optimal Transport, chapter 10.6.3, 10.6.4
		Gromov-Wasserstein Averaging of Kernel and Distance Matrices, Peyre et al ICML 2016
	"""

	def __init__(self, eps, lamda, max_iter, thresh, w_max_iter, w_thresh, w_cost_type='L2', inter_loss_type='square_loss', intra_cost_type='L2', data_precision='double', stable_sinkhorn=False, verbose=False):

		super(Entropic_WGW, self).__init__()

		self.lamda = lamda # balancing between w and gw loss
		self.eps = eps
		self.max_iter = max_iter
		self.thresh = thresh

		self.inter_loss_type = inter_loss_type
		self.intra_cost_type = intra_cost_type
		self.w_cost_type = w_cost_type

		self.verbose = verbose
		self.data_precision = data_precision

		if stable_sinkhorn:
			self.Entropic_W = Stabilized_Entropic_Wasserstein(eps, w_max_iter, w_thresh, verbose=verbose, data_precision=data_precision)
		else:
			self.Entropic_W = Entropic_Wasserstein(eps, w_max_iter, w_thresh, verbose=verbose, data_precision=data_precision)

		self.funcs = self.func_define()

	def forward(self, x, y, px=None, py=None, P=None):
		if px is None:
			px = torch.ones(x.shape[0], device=x.device)
			px /= px.sum()
		if py is None:
			py = torch.ones(y.shape[0], device=y.device)
			py /= py.sum()

		if self.data_precision == 'double':
			x = x.double()
			y = y.double()
			px = px.double()
			py = py.double()            
		else:
			pass

		#t0 = time()
		Cx = self.cost_matrix(x, x, cost_type=self.intra_cost_type)
		Cy = self.cost_matrix(y, y, cost_type=self.intra_cost_type)

		# Cost matrix for compute wasserstein loss
		C = self.cost_matrix(x, y, cost_type=self.w_cost_type)

		#print('\t\tcomputed Cx, Cy, C in %.2f s' %(time()-t0))

		return self.forward_with_cost_matrices(C, Cx, Cy, px, py, P)

	def forward_with_cost_matrices(self, C, Cx, Cy, px, py, P=None):
		
		nx, ny = Cx.shape[0], Cy.shape[0]

		if P is None:
			P = px.unsqueeze(-1) * py.unsqueeze(-2)

		f1, f2, h1, h2 = self.funcs

		#t0 = time()
		Cxy = (f1(Cx) @ px.reshape(-1, 1)).repeat((1, ny)) + (py.reshape(1, -1) @ f2(Cy).T).repeat((nx, 1))
		#print('\t\tcompute Cxy in %.2f s' %(time() - t0))

		for it in range(self.max_iter):

			t1 = time()

			P_old = P

			if self.lamda == 1.:
				L = C

				_, P = self.Entropic_W.forward_with_cost_matrix(L, px, py)

			else:
				L = Cxy - h1(Cx) @ P @ h2(Cy).T
				L2 = 2*L  # Proposition 2 (eq (9)) of Perey et al miss a 2 factor (ref. https://github.com/PythonOT/POT/blob/master/ot/gromov.py)

				LC = self.lamda * C + (1 - self.lamda) * L2  # line 9 of algorithm 3 in 'Graph Optimal Transport for Cross-Domain Alignment'

				_, P = self.Entropic_W.forward_with_cost_matrix(LC, px, py)

			err = torch.norm(P - P_old)
			if err < self.thresh:
				if self.verbose:
					print('Break in Gromov-Wasserstein at %s-th iteration: Err = %f' %(it, err))

				break

			if self.verbose:
				if it % 10 == 0:
					print('Iter: %s | Err = %f' %(it, err))
					print('\t\tP:\n', P.sum(dim=0), P.sum(dim=1), '\n', P)

			#print('\t\tone epoch in forward_with_cost_matrices in %.2fs - err: %s' %(time() - t1, err.item()))
			#print('\t\tP: ', P.sum().item())
			#print('\t\tP:\n', P.sum(dim=0), P.sum(dim=1), '\n', P)


		gw_cost = torch.sum(P * L)

		#print('\t\tgw_cost: ', gw_cost.item())

		return gw_cost, P


	def func_define(self):
		"""
		Define functions f1, f2, h1, h2 to compute the tensor-matrix multiplication as in Proposition 1 of Peyre et al
		"""

		if self.inter_loss_type == 'square_loss':
			
			def f1(a):
				return a**2
			def f2(b):
				return b**2
			def h1(a):
				return a
			def h2(b):
				return 2*b 

		elif self.inter_loss_type == 'kl_loss':

			def f1(a):
				return a * torch.log(a) - a
			def f2(b):
				return b
			def h1(a):
				return a
			def h2(b):
				return torch.log(b)

		else:
			raise NotImplementedError('Inter loss type %s is not implemented!' %(inter_loss_type))

		return f1, f2, h1, h2

	def _cost_matrix(self, x, y, cost_type='L2'):
		"""
		Compute intra cost matrix between each domain
		"""

		if cost_type == 'L2':
			x_row = x.unsqueeze(-2)
			y_col = y.unsqueeze(-3)

			C = torch.sum((x_row - y_col) ** 2, dim=-1)
		else:
			raise NotImplementedError('The cost type %s is not implemented!' %(cost_type))

		return C

	def cost_matrix(self, x, y, cost_type='L2'):
		"""compute pairwise cost matrix by divided into block => reduce memory footprint"""

		n1, n2 = x.shape[-2], y.shape[-2]
		bl = 10
		bl_size = int(n1/bl)

		C = torch.empty((n1, n2), device=x.device)

		x_row = x.unsqueeze(-2)
		y_col = y.unsqueeze(-3)

		for i in range(bl-1):
			#tmp_C = self._cost_matrix(x[i*bl_size: (i+1)*bl_size], y, cost_type)
			#C[i*bl_size: (i+1)*bl_size] = tmp_C
			C[i*(bl_size): (i+1)*bl_size] = torch.sum((x_row[i*bl_size: (i+1)*bl_size] - y_col)**2, dim=-1)

		#tmp_C = self._cost_matrix(x[(bl-1)*bl_size:], y, cost_type)
		#C[(bl-1)*bl_size:] = tmp_C

		C[(bl-1)*bl_size: ] = torch.sum((x_row[(bl-1)*bl_size: ] - y_col)**2, dim=-1)

		return C

	def tensor_matrix_mul(self, Cx, Cy, P):
		"""
		utility used in mul_ot_rescal
		compute the tensor_matrix multiplication in Perey et al 
		"""
		nx, ny = Cx.shape[0], Cy.shape[0]

		f1, f2, h1, h2 = self.funcs

		px, py = P.sum(dim=1), P.sum(dim=0)

		Cxy = (f1(Cx) @ px.reshape(-1, 1)).repeat((1, ny)) + (py.reshape(1, -1) @ f2(Cy).T).repeat((nx, 1))
		L = Cxy - h1(Cx) @ P @ h2(Cy).T

		return L

# reference: https://github.com/ZongxianLee/MMD_Loss.Pytorch/blob/master/mmd_loss.py


import torch
import torch.nn as nn


def cost_matrix(x, y, cost_type='L2'):

    try:
        if cost_type == 'L2':
            x_row = x.unsqueeze(-2)
            y_col = y.unsqueeze(-3)
            C = torch.sum((x_row - y_col)**2, dim=-1)

        else:
            raise NotImplementedError('The cost type %s is not implemented!' %(cost_type))
    except Exception as e:
        nx = int(x.shape[0]/2)

        C = torch.empty(x.shape[0], y.shape[0])

        C[:nx,:nx] = cost_matrix(x[:nx], y[:nx])
        C[nx:,nx:] = cost_matrix(x[nx:], y[nx:])
        C[:nx,nx:] = cost_matrix(x[:nx], y[nx:])
        C[nx:,:nx] = cost_matrix(x[nx:], y[:nx])

    return C


class MMD_loss(nn.Module):

    def __init__(self, kernel_mul = 2.0, kernel_num = 5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None

    def guassian_kernel(self, source, target, n1, n2):

        ns = source.shape[0]
        nt = target.shape[0]

        L2_distance = torch.empty(ns+nt, ns+nt).to(source.device)

        L2_distance[:ns,:ns] = cost_matrix(source, source)
        L2_distance[ns:,ns:] = cost_matrix(target, target)
        L2_distance[:ns,ns:] = cost_matrix(source, target)
        L2_distance[ns:,:ns] = cost_matrix(target, source)


        n_samples = n1 + n2
        kernel_num = self.kernel_num
        kernel_mul = self.kernel_mul
        fix_sigma = self.fix_sigma

        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def forward(self, source, target, n1=None, n2=None):

        if n1 is None and n2 is None:
            n1 = source.shape[0]
            n2 = target.shape[0]

        alpha1 = 1 / (n1**2 - n1)
        alpha2 = 1 / (n2**2 - n2)
        beta = 1 / (n1*n2)

        kernels = self.guassian_kernel(source, target, n1, n2)

        batch_size = int(source.size()[0])
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]

        loss = alpha1*XX.sum() + alpha2*YY.sum() - beta*(XY.sum() + YX.sum())

        return loss
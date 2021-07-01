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

    def __init__(self, kernel_mu=None):
        super(MMD_loss, self).__init__()
        self.kernel_mu = kernel_mu   # tuning mu

    def guassian_kernel(self, source, target, n1, n2): #tuning mu

        ns = source.shape[0]
        nt = target.shape[0]

        L2_distance = torch.empty(ns+nt, ns+nt).to(source.device)

        L2_distance[:ns,:ns] = cost_matrix(source, source)
        L2_distance[ns:,ns:] = cost_matrix(target, target)
        L2_distance[:ns,ns:] = cost_matrix(source, target)
        L2_distance[ns:,:ns] = cost_matrix(target, source)

        return torch.exp(-L2_distance / self.kernel_mu)

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
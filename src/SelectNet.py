import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

eps = np.finfo(float).eps


class SelectLayer(nn.Module):
    def __init__(self, in_dim):
        super(SelectLayer, self).__init__()
        self.in_dim = in_dim
        self.w = nn.Parameter(torch.ones([1, in_dim], dtype=torch.float)*1)
        self.w.requires_grad_(True)

    def forward(self, x):
        batch, _ = x.shape
        ratio = self.calc_ratio() * self.in_dim
        out = ratio.repeat(batch, 1) * x
        return out

    def calc_ratio(self):
        w_prine = torch.sigmoid(self.w)
        w_ratio = w_prine / torch.sum(w_prine)
        return w_ratio

    def compute_loss(self):
        w_prine = torch.sigmoid(self.w)
        w_ratio = w_prine / torch.sum(w_prine)

        w_loss = F.l1_loss(w_prine, torch.zeros_like(w_prine))
        entropy_loss = self.calc_entropy_loss(w_ratio)

        return w_loss, entropy_loss

    def calc_entropy_loss(self, x):
        N = x.shape[0]
        loss = 0
        for i in range(x.shape[-1]):
            a = x[:, i]
            loss += torch.sum(-a*torch.log2(a+eps))
        return loss / N


class SelectNet(nn.Module):
    def __init__(self, in_dim, downstream_model, kernel_weight_list):
        super(SelectNet, self).__init__()
        self.select_lay = SelectLayer(in_dim)
        self.w = self.select_lay.w
        self.downstream_model = downstream_model
        self.kernel_weight_list = kernel_weight_list

    def forward(self, x):
        x = self.select_lay(x)
        y = self.downstream_model(x)
        return y

#     return the regularization loss
    def calc_reg_loss(self, regularization_loss_fn):
        reg_loss = self.calc_kernel_regularization_loss(regularization_loss_fn)
        w_loss, entropy_loss = self.select_lay.compute_loss()

        return reg_loss, w_loss, entropy_loss

    def calc_kernel_regularization_loss(self, loss_fn):
        loss = 0
        for w in self.kernel_weight_list:
            loss += loss_fn(torch.zeros_like(w), w)
        return loss / len(self.kernel_weight_list)

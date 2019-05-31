import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

eps = np.finfo(float).eps


class SelectLayer(nn.Module):
    def __init__(self, in_dim, ver=1, p=1.):
        super(SelectLayer, self).__init__()
        self.ver = ver
        self.p = p
        self.in_dim = in_dim
        self.threshold = 1e-5
        if self.ver in [0]:
            self.w = nn.Parameter(torch.ones([1, in_dim], dtype=torch.float)*1.)
            self.w.requires_grad_(False)
        elif self.ver in [1]:
            self.w = nn.Parameter(torch.ones([1, in_dim], dtype=torch.float)*1.)
            self.w.requires_grad_(True)
        elif self.ver in [2, 4]:
            self.w = nn.Parameter(torch.ones([1, in_dim], dtype=torch.float)*0.1)
            self.w.requires_grad_(False)
        elif self.ver in [3]:
            self.w = nn.Parameter(torch.ones([1, in_dim], dtype=torch.float)*0)
            self.w.requires_grad_(False)
        elif self.ver in [5]:
            self.w = nn.Parameter(torch.ones([1, in_dim], dtype=torch.float)*0)
            self.w.requires_grad_(False)
        else:
            raise NotImplementedError

    def forward(self, x):
        batch, _ = x.shape
        if self.ver in [0]:
            return x
        elif self.ver in [1]:
            out = self.w.repeat(batch, 1) * x
            return out
        elif self.ver in [2, 3, 4]:
            ratio = self.calc_ratio() * self.in_dim
            out = ratio.repeat(batch, 1) * x
            return out
        elif self.ver in [5]:
            out = (self.smooth_sigmoid(self.w).repeat(batch, 1) / 100.) * x
            return out
        else:
            raise NotImplementedError

    def calc_ratio(self):
        if self.ver in [0, 1]:
            w_prine = F.relu(self.w)
            w_ratio = w_prine / torch.sum(w_prine)
        elif self.ver in [2, 4]:
            w_prine = F.relu(self.w)
            w_ratio = w_prine / torch.sum(w_prine)
            w_ratio = torch.clamp(w_ratio-(0.1/self.in_dim), 0, 1)
        elif self.ver in [3]:
            w_prine = self.smooth_sigmoid(self.w)
            w_ratio = w_prine / torch.sum(w_prine)
            w_ratio = torch.clamp(w_ratio-0.01, 0, 1)
        elif self.ver in [5]:
            w_prine = (self.smooth_sigmoid(self.w) / 100.)
            w_ratio = w_prine / torch.sum(w_prine)
        else:
            raise NotImplementedError
        return w_ratio

    def compute_loss(self):
        if self.ver in [0]:
            # ret 0
            w_loss = 0*torch.mean(self.w)
            entropy_loss = 0*w_loss
            return w_loss, entropy_loss
        elif self.ver in [1]:
            w_loss = self.w.norm(self.p) / self.in_dim
            entropy_loss = 0*w_loss
            return w_loss, entropy_loss
        elif self.ver in [2]:
            w_loss = F.relu(self.w).norm(self.p) / self.in_dim
            entropy_loss = self.calc_entropy_loss(self.calc_ratio())
            return w_loss, entropy_loss
        elif self.ver in [3]:
            w_loss = self.smooth_sigmoid(self.w).norm(self.p) / self.in_dim
            entropy_loss = self.calc_entropy_loss(self.calc_ratio())
            return w_loss, entropy_loss
        elif self.ver in [4]:
            ratio = self.calc_ratio()
            w_loss = ratio.norm(self.p) / self.in_dim
            entropy_loss = self.calc_entropy_loss(ratio)
            return w_loss, entropy_loss
        elif self.ver in [5]:
            w_loss = self.smooth_sigmoid(self.w).norm(self.p) / self.in_dim
            entropy_loss = 0*w_loss
            return w_loss, entropy_loss
        else:
            raise NotImplementedError

    def calc_entropy_loss(self, x):
        N = x.shape[0]
        loss = 0
        for i in range(x.shape[-1]):
            a = x[:, i]
            loss += torch.sum(-a*torch.log2(a+eps))
        return loss / N

    def clip_w(self):
        if self.ver in [0, 1, 5]:
            pass
        elif self.ver in [2, 4]:
            self.w.data.clamp_(self.threshold, 100)
        elif self.ver in [3]:
            pass
            self.w.data.clamp_(-10, 10)
        else:
            raise NotImplementedError

    def activate_w(self):
        if not self.w.requires_grad:
            self.w.requires_grad_(True)

    def smooth_sigmoid(self, x):
        return 1. / (1. + 0.4*torch.exp(-x))


class SelectNet(nn.Module):
    def __init__(self, in_dim, downstream_model, kernel_weight_list, ver, p, use_norm_lay):
        super(SelectNet, self).__init__()
        self.norm = nn.BatchNorm1d(in_dim) if use_norm_lay else False
        self.select_lay = SelectLayer(in_dim, ver, p)
        self.w = self.select_lay.w
        self.clip_w = self.select_lay.clip_w
        self.activate_w = self.select_lay.activate_w
        self.downstream_model = downstream_model
        self.kernel_weight_list = kernel_weight_list

    def forward(self, x):
        if self.norm:
            x = self.norm(x)
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

import torch.nn as nn


class Autoendecoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, depth, act_fn):
        super(Autoendecoder, self).__init__()
        assert depth > 0
        self.act_fn = act_fn
        self.en = nn.ModuleList()
        self.de = nn.ModuleList()
        self.en.append(nn.Linear(in_dim, hidden_dim))
        self.de.append(nn.Linear(hidden_dim, in_dim))
        for i in range(depth-1):
            self.en.append(nn.Linear(hidden_dim, hidden_dim))
            self.de.append(nn.Linear(hidden_dim, hidden_dim))
#         for regularizer
        self.kernel_weights = []
        for linear in self.en:
            self.kernel_weights.append(linear.weight)
        for linear in self.de:
            self.kernel_weights.append(linear.weight)

    def forward(self, x):
        for lay in self.en:
            x = self.act_fn(lay(x))
        for lay in self.de[:-1]:
            x = self.act_fn(lay(x))
        out = self.de[-1](x)
        return out

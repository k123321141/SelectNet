import torch.nn as nn


class SimpleDNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, depth, act_fn):
        super(SimpleDNN, self).__init__()
        assert depth > 0
        self.act_fn = act_fn
        self.linears = nn.ModuleList()
        if depth == 1:
            self.linears.append(nn.Linear(in_dim, out_dim))
        else:
            self.linears.append(nn.Linear(in_dim, hidden_dim))
        for i in range(depth-1):
            if i == depth-2:
                self.linears.append(nn.Linear(hidden_dim, out_dim))
            else:
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))

#         for regularizer
        self.kernel_weights = []
        for linear in self.linears:
            self.kernel_weights.append(linear.weight)

    def forward(self, x):
        for lay in self.linears[:-1]:
            x = self.act_fn(lay(x))
        out = self.linears[-1](x)
        return out

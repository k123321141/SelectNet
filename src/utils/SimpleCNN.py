import torch.nn as nn


class SimpleCNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, depth):
        super(SimpleCNN, self).__init__()
        assert depth > 0
        self.cnns = nn.ModuleList()
        self.kernel_weights = []
        cnn = nn.Conv2d(in_dim, hidden_dim, kernel_size=5, stride=1, padding=2)
        self.kernel_weights.append(cnn.weight)
        self.cnns.append(nn.Sequential(
                cnn,
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),
                ))
        for i in range(depth-1):
            cnn = nn.Conv2d(hidden_dim*2**i, hidden_dim*2**(i+1), kernel_size=3, stride=1, padding=1)
            self.kernel_weights.append(cnn.weight)
            self.cnns.append(nn.Sequential(
                    cnn,
                    nn.BatchNorm2d(hidden_dim*2**(i+1)),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2),
                    ))

    def forward(self, x):
        for lay in self.cnns:
            x = lay(x)
        return x

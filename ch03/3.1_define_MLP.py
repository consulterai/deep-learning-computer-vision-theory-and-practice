import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 16 * 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16 * 16, 10)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        # N 1 28 28
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        out = self.sig(x)
        return out

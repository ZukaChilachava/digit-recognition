
import torch.nn as nn


class MultiClassModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MultiClassModel, self).__init__()
        self.rl = nn.ReLU()
        self.L1 = nn.Linear(input_size, 48)
        self.L2 = nn.Linear(48, 32)
        self.output = nn.Linear(32, num_classes)

    def forward(self, x):
        ReLU1 = self.rl(self.L1(x))
        ReLU2 = self.rl(self.L2(ReLU1))

        return self.output(ReLU2)

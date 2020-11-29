import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, item_count):
        super(Discriminator, self).__init__()
        self.dis = nn.Sequential(
            nn.Linear(item_count * 2, 125),

            nn.ReLU(True),

            nn.Linear(125, 1),
            nn.Sigmoid()
        )

    def forward(self, data, condition):
        result = self.dis(torch.cat((data, condition), 1))
        return result


class Generator(nn.Module):
    def __init__(self, item_count):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            nn.Linear(item_count, 400),
            nn.ReLU(True),

            nn.Linear(400, 400),
            nn.ReLU(True),
            nn.Linear(400, 400),
            nn.ReLU(True),
            nn.Linear(400, 400),
            nn.ReLU(True),
            nn.Linear(400, item_count),
            nn.Sigmoid()
        )

    def forward(self, x):
        result = self.gen(x)
        return result

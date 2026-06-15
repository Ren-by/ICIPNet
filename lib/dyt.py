import torch
import torch.nn as nn

import torch
import torch.nn as nn


class DyT(nn.Module):

    def __init__(self, C, init_alpha=1.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1) * init_alpha)  # scalar
        self.beta = nn.Parameter(torch.ones(1, C, 1, 1))  # [1,C,1,1]
        self.gamma = nn.Parameter(torch.zeros(1, C, 1, 1))  # [1,C,1,1]
        self.act = nn.Tanh()

    def forward(self, x):  # x: [B,C,H,W]
        x = self.act(x * self.alpha)  # 缩放 + 激活
        x = self.beta * x + self.gamma  # 逐通道仿射变换
        return x

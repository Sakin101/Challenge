import torch
import torch.distributions.binomial
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

torch.manual_seed(1)

class MultiFrameModel(nn.Module):
    def __init__(self, backbone, linear_dim, num_of_frames, drop_p=0.4):
        super().__init__()
        self.backbone = backbone
        self.linear = nn.Linear(linear_dim * num_of_frames, 3)
        self.drop_p = drop_p

        # self.remaining_binomial = torch.distributions.binomial.Binomial(num_of_frames-1, torch.tensor([1-drop_p]))
    
    def forward(self, x):
        B, F, C, H, W = x.shape

        x = x.view(B*F, C, H, W)
        x = self.backbone(x)
        x = x.flatten(1)
        x = x.view(B, F, -1)
        if self.training:
            b = torch.bernoulli(torch.ones(B, F-1)*(1-self.drop_p)).to('cuda')
            x[:,:-1,:] *= b.view(B, F-1, 1).repeat(1, 1, x.shape[2])
        else:
            x[:,:-1,:] *= (1-self.drop_p)
        x = x.view(B, F * x.shape[-1])
        x = self.linear(x)

        return x

        # num_of_frames_remaining = self.binomial.sample()
        # indices = torch.ones(B, F-1)
        # frame_indices = indices.multinomial(num_of_frames_remaining, replacement=False)
        # frames = 

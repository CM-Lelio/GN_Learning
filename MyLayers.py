# -.- encoding: utf-8 -.-

# using torch.nn to build a network
import torch.nn as nn
import torch

class NoiseLayer1d(nn.Module):
    # this is the initiaion
    def __init__(self, alpha=0):
        # inherit the initiation of parent(super) class
        super(NoiseLayer1d, self).__init__()
        self.alpha = alpha

    def forward(self, input):
        noise = torch.randn(input.size())
        #temp_input = input.view(input.size()[0],-1)
        stds = input.std(1).unsqueeze(-1)
        noise = noise * stds * self.alpha
        output = input + noise
        return output

class NoiseLayer2d(nn.Module):
    # this is the initiaion
    def __init__(self, alpha=0):
        # inherit the initiation of parent(super) class
        super(NoiseLayer2d, self).__init__()
        self.alpha = alpha

    def forward(self, input):
        noise = torch.randn(input.size())
        #temp_input = input.view(input.size()[0],-1)
        stds = input.std(2).std(2).unsqueeze(-1).unsqueeze(-1)
        noise = noise * stds * self.alpha
        output = input + noise
        return output

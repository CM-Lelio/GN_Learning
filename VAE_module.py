import torch 
import torch.nn as nn
from torch.autograd import Variable

class ConvNet(nn.Module):
    def __init__(self, out_dim=32):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(                                # 1@28x28
            nn.Conv2d(1, 16, kernel_size=4, stride=2, padding=1),   # 16@14x14
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),  # 32@7x7
            nn.BatchNorm2d(32),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2),             # 64@3x3
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, out_dim, kernel_size=3, stride=1, padding=1),  # 32@3x3
            nn.BatchNorm2d(out_dim),
            nn.ReLU())
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        return out

class MLP(nn.Module):
    def __init__(self, in_dim=32, hidden_dim=32, out_dim=1):
        super(MLP, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim))

    def forward(self, x):
        # out = self.gap(x)
        out = x
        out = out.view(out.size()[0],-1)
        out = self.fc(out)
        return out


class ConvTransNet(nn.Module):
    def __init__(self, in_dim=32):
        super(ConvTransNet, self).__init__()
        self.layer1 = nn.Sequential(                                            # 32@3x3
            nn.ConvTranspose2d(in_dim, 64, kernel_size=3, stride=1, padding=1),# 64@3x3
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2),                # 32@7x7
            nn.BatchNorm2d(32),
            nn.ReLU(),
            )
        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),     # 16@14x14
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1),      # 1@28x28
            nn.Sigmoid(),
            )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        return out

class VAENet(nn.Module):
    def __init__(self):
        super(VAENet, self).__init__()
        self.encoder = ConvNet()
        self.meanGenerator = MLP(32*3*3)
        self.varGenerator = MLP(32*3*3)
        self.decoder = ConvTransNet()

    def sampling(self, mu, log_var, size_Z):
        batch_m = size_Z[0]
        dim_z = 1
        for i in range(1, len(size_Z)):
            dim_z *= size_Z[i]
        eps = Variable(torch.randn(batch_m, dim_z))
        if mu.is_cuda:
            eps = eps.cuda()
        return mu + torch.exp(log_var / 2) * eps

    def forward(self, x):
        x_code = self.encoder(x)
        x_mu = self.meanGenerator(x_code)
        x_var = self.varGenerator(x_code)
        x_sample = self.sampling(x_mu, x_var, x_code.size())
        x_sample = x_sample.view(x_code.size())
        x_out = self.decoder(x_sample)
        kl_loss = torch.mean(0.5 * torch.sum(torch.exp(x_var) + x_mu**2 - 1. - x_var, 1))
        return x_out, kl_loss

class VAENet_small(nn.Module):
    def __init__(self):
        super(VAENet_small, self).__init__()
        self.encoder = MLP(1*28*28,64,32)
        self.meanGenerator = MLP(32,8,1)
        self.varGenerator = MLP(32,8,1)
        self.decoder = nn.Sequential(
            MLP(32,64,1*28*28),
            nn.Sigmoid(),
            )

    def sampling(self, mu, log_var, size_Z):
        batch_m = size_Z[0]
        dim_z = 1
        for i in range(1, len(size_Z)):
            dim_z *= size_Z[i]
        eps = Variable(torch.randn(batch_m, dim_z))
        if mu.is_cuda:
            eps = eps.cuda()
        return mu + torch.exp(log_var / 2) * eps

    def forward(self, x):
        x_view = x.view(x.size()[0],-1)
        x_code = self.encoder(x_view)
        x_mu = self.meanGenerator(x_code)
        x_var = self.varGenerator(x_code)
        x_sample = self.sampling(x_mu, x_var, x_code.size())
        x_sample = x_sample.view(x_code.size())
        x_out = self.decoder(x_sample)
        kl_loss = torch.mean(0.5 * torch.sum(torch.exp(x_var) + x_mu**2 - 1. - x_var, 1))
        return x_out, kl_loss
# -.- encoding: utf-8 -.-

# using torch.nn to build a network
import torch.nn as nn
import torch

from MyLayers import *

## baseline
class BaseModel(nn.Module):
    # this is the initiaion
    def __init__(self):
        # inherit the initiation of parent(super) class
        super(BaseModel, self).__init__()
        
        # define a cnn
        self.conv1=nn.Sequential(    # 28x28
            nn.Conv2d(3, 8, 3, 1, 1),  # 28x28
            nn.BatchNorm2d(8),
            nn.ReLU()
        )

        self.Pool2d = nn.MaxPool2d(2)
        
        self.conv2=nn.Sequential(
            nn.Conv2d(8, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        self.conv3=nn.Sequential(
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(32, 32, 3, 2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.3),
            )     

        self.linear=nn.Sequential(  
            nn.Linear(32*3*3, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        
        self.calssifier=nn.Sequential(  
            nn.Linear(32, 10)
        )

    def forward(self, input, flagTest=False):
        temp_out = self.conv1(input)        # 28x28
        temp_out = self.Pool2d(temp_out)    # 14x14
        temp_out = self.conv2(temp_out)     # 14x14
        temp_out = self.Pool2d(temp_out)    # 7x7
        temp_out = self.conv3(temp_out)     # 3x3
        
        temp_out = temp_out.view(temp_out.size()[0],-1)    # 288
        
        temp_out = self.linear(temp_out)     
        output = self.calssifier(temp_out)    # 10:class
        
        if flagTest:
            return output,temp_out
        else: 
            return output
            
## NoiseNet
class BaseModel_Noise(nn.Module):
    # this is the initiaion
    def __init__(self):
        # inherit the initiation of parent(super) class
        super(BaseModel_Noise, self).__init__()
        
        # define a cnn
        self.conv1=nn.Sequential(    # 28x28
            nn.Conv2d(3, 8, 3, 1, 1),  # 28x28
            nn.BatchNorm2d(8),
            nn.ReLU()
        )

        self.Pool2d = nn.MaxPool2d(2)
        self.Noise1d = NoiseLayer1d(0.1)
        self.Noise2d = NoiseLayer2d(0.1)
        
        self.conv2=nn.Sequential(
            nn.Conv2d(8, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        self.conv3=nn.Sequential(
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(32, 32, 3, 2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.3),
            )     

        self.linear=nn.Sequential(  
            nn.Linear(32*3*3, 64),
            nn.ReLU(),
            NoiseLayer1d(0.1),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        
        self.calssifier=nn.Sequential(  
            nn.Linear(32, 10)
        )

    def forward(self, input, flagTest=False):
        temp_out = self.conv1(input)        # 28x28
        temp_out = self.Noise2d(temp_out)
        temp_out = self.Pool2d(temp_out)    # 14x14
        temp_out = self.conv2(temp_out)     # 14x14
        temp_out = self.Noise2d(temp_out)
        temp_out = self.Pool2d(temp_out)    # 7x7
        temp_out = self.conv3(temp_out)     # 3x3
        temp_out = self.Noise2d(temp_out)
        
        temp_out = temp_out.view(temp_out.size()[0],-1)    # 288
        
        temp_out = self.linear(temp_out)    
        output = self.calssifier(temp_out)    # 10:class
        
        if flagTest:
            return output,temp_out
        else: 
            return output

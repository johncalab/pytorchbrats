import torch
import torch.nn as nn
import torch.nn.functional as F

# a simple encoderdecoder-type model
class Crush(nn.Module):
    def __init__(self,D=32,S=64,C=4,crush_size=32):
        super(Crush,self).__init__()
        
        self.dimensions = (C,D,D,S)
        self.in_features = int(D*D*S*C)
        self.crush = crush_size
        self.out_features = int(D*D*S)
        self.enc = nn.Linear(in_features=self.in_features,out_features=self.crush,bias=True)
        self.act = nn.ReLU()
        self.dec = nn.Linear(in_features=self.crush,out_features=self.out_features)
        self.sig = nn.Sigmoid()
    
    def forward(self, x_in):
        batch_len = x_in.shape[0]
        x = x_in.view(batch_len,-1)
        x = self.enc(x)
        x = self.act(x)
        x = self.dec(x)
        
        dummy_dim = (-1,) + self.dimensions[1:]
        x_out = x.view(dummy_dim)
        return x_out

# a simple sequence of convolutional layers
class ConvSeq(nn.Module):
    def __init__(self,input_channels=4):
        super(ConvSeq,self).__init__()
        
        self.input_channels = input_channels
        
        self.c1 = self.ConvLayer(in_channels=self.input_channels)
        self.c2 = self.ConvLayer()
        self.c3 = self.ConvLayer()
        self.cfinal = self.ConvLayer(out_channels=1, relu=False)
    
    def forward(self, x_in):
        x = self.c1(x_in)
        x = self.c2(x)
        x = self.c3(x)
        x_out = self.cfinal(x).squeeze(1)
            
        return x_out
    
    def ConvLayer(self, in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1,
        bias=True, relu=True, batchnorm=True, dropout=False, p=0.5):
        layer = nn.Sequential()
        if dropout:
            layer.add_module('dropout',nn.Dropout(p=p))
        conv = nn.Conv3d(in_channels=in_channels,
                         out_channels=out_channels,
                         kernel_size=kernel_size,
                         padding=padding,
                         bias=bias)
        layer.add_module('conv',conv)
        if relu:
            layer.add_module('relu',nn.ReLU())
        if batchnorm:
            layer.add_module('batchnorm',nn.BatchNorm3d(out_channels))

        return layer
import torch
import torch.nn as nn
import torch.nn.functional as F
"""
Contains a few models to be used for training.
NOTE: forward pass never applies sigmoid (or .round).
This is in case you use BCELossWithLogits, or similar.
(such loss have an optimized version of loss circ sigmoid)

    Crush
    ConvSeq
    Small3dUcat
    Small3dUadd
    UU3d
    Unet3d

"""

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

# small 3d u-net with concatenating skip connection
class Small3dUcat(nn.Module):
    def __init__(self,input_channels=4,num_filters=32):
        super(Small3dUcat,self).__init__()        
        
        self.input_channels = input_channels
        self.num_filters = num_filters
        
        self.c1 = self.ConvLayer(in_channels=self.input_channels,out_channels=num_filters)
        self.c2 = self.ConvLayer(in_channels=num_filters,out_channels=num_filters)

        self.c3 = self.ConvLayer(in_channels=num_filters,out_channels=num_filters)
        self.c4 = self.ConvLayer(in_channels=num_filters,out_channels=num_filters)
        
        self.c5 = self.ConvLayer(in_channels=2*num_filters,out_channels=num_filters)
        self.c6 = self.ConvLayer(in_channels=num_filters,out_channels=num_filters)
        self.c7 = self.ConvLayer(in_channels=num_filters,out_channels=1)
    
    def forward(self, x_in, evaluating=False):
        x1 = self.c1(x_in)
        x1 = self.c2(x1)
        
        x2 = F.max_pool3d(x1,kernel_size=2)
        x2 = self.c3(x2)
        x2 = self.c4(x2)
        
        x2 = F.interpolate(x2, scale_factor=2)
        # concatenate x1,x2
        x = torch.cat([x1,x2],dim=1)
        x = self.c5(x)
        x = self.c6(x)
        x = self.c7(x)
        
        x_out = x.squeeze(1)

        return x_out
    
    def ConvLayer(self, in_channels=32, out_channels=32, kernel_size=3,
                  stride=1, padding=1, bias=True, relu=True, batchnorm=True):
        layer = nn.Sequential()
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

# small 3d u-net with addition skip connection
class Small3dUadd(nn.Module):
    def __init__(self,input_channels=4,num_filters=32):
        super(Small3dUadd,self).__init__()
        
        self.input_channels = input_channels
        self.num_filters = num_filters
        
        self.c1 = self.ConvLayer(in_channels=self.input_channels,out_channels=num_filters)
        self.c2 = self.ConvLayer(in_channels=num_filters,out_channels=num_filters)

        self.c3 = self.ConvLayer(in_channels=num_filters,out_channels=num_filters)
        self.c4 = self.ConvLayer(in_channels=num_filters,out_channels=num_filters)
        
        self.c5 = self.ConvLayer(in_channels=num_filters,out_channels=num_filters)
        self.c6 = self.ConvLayer(in_channels=num_filters,out_channels=num_filters)
        self.c7 = self.ConvLayer(in_channels=num_filters,out_channels=1)
    
    def forward(self, x_in, evaluating=False):
        x1 = self.c1(x_in)
        x1 = self.c2(x1)
        
        x2 = F.max_pool3d(x1,kernel_size=2)
        x2 = self.c3(x2)
        x2 = self.c4(x2)
        
        x2 = F.interpolate(x2, scale_factor=2)
        # add x1,x2
        x = F.relu(x1+x2)
        x = self.c5(x)
        x = self.c6(x)
        x = self.c7(x)
        
        x_out = x.squeeze(1)

        return x_out
    
    def ConvLayer(self, in_channels=32, out_channels=32, kernel_size=3,
                  stride=1, padding=1, bias=True, relu=True, batchnorm=True):
        layer = nn.Sequential()
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

# 3d UU-net
class UU3d(nn.Module):
    def __init__(self,input_channels=4, num_filters=16):
        super(UU3d,self,).__init__()

        n = num_filters
        
        self.c0 = self.ConvLayer(in_channels=input_channels,out_channels=n,
                                 kernel_size=3,stride=1,padding=1)

        self.d1 = self.ConvLayer(in_channels=n,out_channels=2*n,
                                 stride=2,kernel_size=2,padding=0)
        self.c1 = self.ConvLayer(in_channels=2*n,out_channels=2*n,
                                kernel_size=3,stride=1,padding=1)

        self.d2 = self.ConvLayer(in_channels=2*n,out_channels=4*n,
                                 stride=2,kernel_size=2,padding=0)
        self.c2 = self.ConvLayer(in_channels=4*n,out_channels=n*4,
                                kernel_size=3,stride=1,padding=1)

        self.d3 = self.ConvLayer(in_channels=n*4,out_channels=n*8,
                                 stride=2,kernel_size=2,padding=0)
        self.c3 = self.ConvLayer(in_channels=n*8,out_channels=n*8,
                                kernel_size=3,stride=1,padding=1)
        
        self.f1 = self.ConvLayer(in_channels=n*8,out_channels=n*4,
                                kernel_size=1,stride=1,padding=0)
        
        self.u1 = self.ConvLayer(in_channels=n*4,out_channels=n*4,
                                kernel_size=2,stride=2,padding=0,transpose=True)
        self.c4 = self.ConvLayer(in_channels=n*8,out_channels=n*4,
                                 kernel_size=3,stride=1,padding=1)
        self.f2 = self.ConvLayer(in_channels=n*4,out_channels=n*2,
                                kernel_size=1,stride=1,padding=0)
        
        self.u2 = self.ConvLayer(in_channels=n*2,out_channels=n*2,
                                kernel_size=2,stride=2,padding=0,transpose=True)
        self.c5 = self.ConvLayer(in_channels=n*4,out_channels=n*2,
                                kernel_size=3,stride=1,padding=1)
        self.f3 = self.ConvLayer(in_channels=n*2,out_channels=n,
                                kernel_size=1,stride=1,padding=0)
        
        self.u3 = self.ConvLayer(in_channels=n,out_channels=n,
                                kernel_size=2,stride=2,padding=0,transpose=True)
        self.c6 = self.ConvLayer(in_channels=n*2,out_channels=n*2,
                                kernel_size=3,stride=1,padding=1)
        
        self.f4 = self.ConvLayer(in_channels=n*4,out_channels=n*2,
                                kernel_size=1,stride=1,padding=0)
        self.f5 = self.ConvLayer(in_channels=n*2,out_channels=n,
                                kernel_size=1,stride=1,padding=0)
        self.f6 = self.ConvLayer(in_channels=n*2,out_channels=1,
                                kernel_size=1,stride=1,padding=0)
        
        self.u4 = self.ConvLayer(in_channels=n*2,out_channels=n,
                                kernel_size=2,stride=2,padding=0,transpose=True)
        self.u5 = self.ConvLayer(in_channels=n,out_channels=1,
                                kernel_size=2,stride=2,padding=0,transpose=True)
        
        self.act = nn.PReLU(num_parameters=1)  

    def forward(self, x_in, evaluating=False):
        x0 = self.c0(x_in)
        x1 = self.d1(x0)
        x2 = self.c1(x1)
        x3 = self.d2(x2+x1)
        x4 = self.c2(x3)
        x5 = self.d3(x4+x3)
        x6 = self.c3(x5)
        x7 = self.f1(x6+x5)
        x8 = self.u1(x7)
        x9 = self.c4(torch.cat([x8,x4],dim=1))
        x10 = self.f2(x9)
        x11 = self.u2(x10)
        x12 = self.c5(torch.cat([x11,x2],dim=1))
        x13 = self.f3(x12)
        x14 = self.u3(x13)
        x15 = self.f4(x9)
        x16 = self.u4(x15)
        x17 = self.f5(x12)
        # x18 is missing, the graph I sketched had a box I did not use
        x19 = self.u5(x16+x17)
        x20 = self.c6(torch.cat([x14,x0],dim=1))
        x21 = self.f6(x20)
        x22 = self.act(x21+x19)

        x_out = x22.squeeze(1)

        return x_out

    def ConvLayer(self,in_channels,out_channels,kernel_size,
                  stride,padding,dilation=1,bias=True,prelu=True,batchnorm=True,transpose=False):
        layer = nn.Sequential()
        if transpose:
            tconv = nn.ConvTranspose3d(in_channels=in_channels,
                             out_channels=out_channels,
                             kernel_size=kernel_size,
                             stride=stride,
                             padding=padding,
                             dilation=dilation,
                             bias=bias)
            layer.add_module('tconv',tconv)
        else:
            conv = nn.Conv3d(in_channels=in_channels,
                             out_channels=out_channels,
                             kernel_size=kernel_size,
                             stride=stride,
                             padding=padding,
                             dilation=dilation,
                             bias=bias)
            layer.add_module('conv',conv)

        if batchnorm:
            layer.add_module('batchnorm',nn.BatchNorm3d(num_features=out_channels))

        if prelu:
            layer.add_module('prelu',nn.PReLU(num_parameters=out_channels))

        return layer

# a proper U-net
# include picture of architecture
# small 3d u-net with addition skip connection
class Unet3d(nn.Module):
    def __init__(self,input_channels=4,num_filters=16):
        super(Unet3d,self).__init__()
        
        # Structure:
        # cccmcccmcccmcccmcccucccucccucccucccf
        
        self.input_channels = input_channels
        c = self.input_channels
        self.num_filters = num_filters
        n = self.num_filters
        
        # down
        self.c1 = self.ConvLayer(in_channels=c,out_channels=n)
        self.c2 = self.ConvLayer(in_channels=n,out_channels=n)
        self.c3 = self.ConvLayer(in_channels=n,out_channels=n)

        self.c4 = self.ConvLayer(in_channels=n,out_channels=2*n)
        self.c5 = self.ConvLayer(in_channels=2*n,out_channels=2*n)
        self.c6 = self.ConvLayer(in_channels=2*n,out_channels=2*n)

        self.c7 = self.ConvLayer(in_channels=2*n,out_channels=4*n)
        self.c8 = self.ConvLayer(in_channels=4*n,out_channels=4*n)
        self.c9 = self.ConvLayer(in_channels=4*n,out_channels=4*n)

        self.c10 = self.ConvLayer(in_channels=4*n,out_channels=8*n)
        self.c11 = self.ConvLayer(in_channels=8*n,out_channels=8*n)
        self.c12 = self.ConvLayer(in_channels=8*n,out_channels=8*n)

        self.c13 = self.ConvLayer(in_channels=8*n,out_channels=16*n)
        self.c14 = self.ConvLayer(in_channels=16*n,out_channels=16*n)
        self.c15 = self.ConvLayer(in_channels=16*n,out_channels=16*n)
        
        # up
        self.c16 = self.ConvLayer(in_channels=16*n+8*n,out_channels=8*n)
        self.c17 = self.ConvLayer(in_channels=8*n,out_channels=8*n)
        self.c18 = self.ConvLayer(in_channels=8*n,out_channels=8*n)

        
        self.c19 = self.ConvLayer(in_channels=8*n+4*n,out_channels=4*n)
        self.c20 = self.ConvLayer(in_channels=4*n,out_channels=4*n)
        self.c21 = self.ConvLayer(in_channels=4*n,out_channels=4*n)

        
        self.c22 = self.ConvLayer(in_channels=4*n+2*n,out_channels=2*n)
        self.c23 = self.ConvLayer(in_channels=2*n,out_channels=2*n)
        self.c24 = self.ConvLayer(in_channels=2*n,out_channels=2*n)

        self.c25 = self.ConvLayer(in_channels=2*n+n,out_channels=n)
        self.c26 = self.ConvLayer(in_channels=n,out_channels=n)
        self.c27 = self.ConvLayer(in_channels=n,out_channels=n)
        
        self.f = self.ConvLayer(in_channels=n,out_channels=1,
                                kernel_size=1,padding=0)

    def forward(self, x_in, evaluating=False):
        x = self.c1(x_in)
        x = self.c2(x)
        x3 = self.c3(x)
        
        x = F.max_pool3d(x3,kernel_size=2)
        x = self.c4(x)
        x = self.c5(x)
        x6 = self.c6(x)
        
        x = F.max_pool3d(x6,kernel_size=2)
        x = self.c7(x)
        x = self.c8(x)
        x9 = self.c9(x)
        
        x = F.max_pool3d(x9,kernel_size=2)
        x = self.c10(x)
        x = self.c11(x)
        x12 = self.c12(x)
        
        x = F.max_pool3d(x12,kernel_size=2)
        x = self.c13(x)
        x = self.c14(x)
        x = self.c15(x)
        
        x = F.interpolate(x, scale_factor=2)
        x = torch.cat([x,x12],dim=1)
        x = self.c16(x)
        x = self.c17(x)
        x = self.c18(x)
        
        x = F.interpolate(x,scale_factor=2)
        x = torch.cat([x,x9],dim=1)
        x = self.c19(x)
        x = self.c20(x)
        x = self.c21(x)
        
        x = F.interpolate(x,scale_factor=2)
        x = torch.cat([x,x6],dim=1)
        x = self.c22(x)
        x = self.c23(x)
        x = self.c24(x)
        
        x = F.interpolate(x,scale_factor=2)
        x = torch.cat([x,x3],dim=1)
        x = self.c25(x)
        x = self.c26(x)
        x = self.c27(x)
        
        x = self.f(x)
        x_out = x.squeeze(1)

        return x_out
            
    def ConvLayer(self, in_channels=32, out_channels=32, kernel_size=3,
                  stride=1, padding=1, bias=True, relu=True, batchnorm=True):
        layer = nn.Sequential()
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
import torch
import torch.nn as nn
import torch.nn.functional as F
class Model(torch.nn.Module):
    def __init__(self,filters=8):
        super(Model,self).__init__()
        self.dlayer1 = self.squash(1,filters,padding=1)
        self.dlayer2 = self.squash(filters,filters,padding=1)
        self.down1 = nn.MaxPool3d(2)
        
        self.dlayer3 = self.squash(filters,filters,padding=1)
        self.dlayer4 = self.squash(filters,filters,padding=1)
        self.down2 = nn.MaxPool3d(2)
        
        self.flayer1 = self.squash(filters,filters,padding=1)
        self.flayer2 = self.squash(filters,filters,padding=1)
        self.up1 = nn.Upsample(scale_factor=2)
        
        # here we concatenate x5 and x9
        self.ulayer4 = self.grow(2*filters,filters,padding=1)
        self.ulayer3 = self.grow(filters,filters,padding=1)
        self.up2 = nn.Upsample(scale_factor=2)
        
        # here we connect x2 and x12
        self.ulayer2 = self.grow(2*filters,filters,padding=1)
        self.ulayer1 = self.grow(filters,1,padding=1)
        

    def forward(self, x):
        x0 = x
        x1 = self.dlayer1(x0)
        x2 = self.dlayer2(x1)
        x3 = self.down1(x2)
        x4 = self.dlayer3(x3)
        x5 = self.dlayer4(x4)
        x6 = self.down2(x5)
        x7 = self.flayer1(x6)
        x8 = self.flayer2(x7)
        x9 = self.up1(x8)

        # here we concatenate x5 and x9
        w = torch.cat([x5,x9],dim=1)
        x10 = self.ulayer4(w)
        x11 = self.ulayer3(x10)
        x12 = self.up2(x11)
        
        # here we connect x2 and x12
        z = torch.cat([x2,x12],dim=1)
        x13 = self.ulayer2(z)
        x14 = self.ulayer1(x13)
        
        y = x14
        return y

        
    def squash(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, bias=False):
        return nn.Sequential( nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride,padding=padding,bias=bias), nn.ReLU() )
    
    def grow(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, bias=False):
        return nn.Sequential( nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride,padding=padding,bias=bias), nn.ReLU() )

def dice_loss(input, target):
    input = torch.sigmoid(input)
    smooth = 1.0

    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    
    return ((2.0 * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))

# might actually not need this
class MyLoss (nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        loss = dice_loss(input,target)
        return loss
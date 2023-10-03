import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import logging
from multiprocessing import Pool
from tqdm import tqdm
from functools import partial
import numpy as np
from PIL import Image
import torch.nn.init as init
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.mpconv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels//2, in_channels//2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class ResUNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResUNet, self).__init__()
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.up1 = Up(384, 64)
        
        '''self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256)
        self.up2 = Up(512, 128)
        self.up3 = Up(256, 64)
        self.up4 = Up(128, 64)'''
        self.outc = OutConv(64, out_channels)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        '''x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)'''
        x = self.up1(x3, x2)
        x = self.outc(x)
        return x
        

# You can then use this model for tasks like image segmentation by passing input data through it.
# Be sure to adjust the input and output channels and train the model as needed for your specific task.


# Define Attention Block
class AttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(AttentionBlock, self).__init__()
        self.convolution = nn.Conv2d(in_channels, 1, kernel_size=1)

    def forward(self, x):
        attention = self.convolution(x)
        attention_weights = torch.sigmoid(attention)
        x = x * attention_weights
        return x

# Define Attention U-Net
class AttentionUNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AttentionUNet, self).__init__()
        # Encoder
        self.encoder_conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.encoder_conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        
        # Attention Block
        self.attention = AttentionBlock(64)
        
        # Decoder
        self.decoder_conv1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.decoder_conv2 = nn.Conv2d(64, out_channels, kernel_size=3, padding=1)
        
    def forward(self, x):
        # Encoder
        x = F.interpolate(x, size=(128, 128), mode='bilinear', align_corners=True)
        x1 = self.encoder_conv1(x)
        x2 = self.encoder_conv2(x1)
        
        # Attention Block
        attention = self.attention(x2)
        
        # Decoder
        x3 = self.decoder_conv1(attention)
        x4 = self.decoder_conv2(x3)
        return x4
        


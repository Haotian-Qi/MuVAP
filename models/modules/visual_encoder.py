##
# ResNet18 Pretrained network to extract lip embedding
# This code is modified based on https://github.com/lordmartian/deep_avsr
##

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cpc_encoder import CConv1d


class VisualEncoder(nn.Module):
    #: Width of the final `visualConv1D`. Fixed by the architecture, so the
    #: cross-attention stack has to be this wide too.
    OUT_DIM = 256

    def __init__(self):
        super().__init__()
        self.out_dim = self.OUT_DIM
        self.frontend = visualFrontend()
        self.tcn = visualTCN()
        self.conv1d = visualConv1D()

        # Register normalization constants as buffers
        self.register_buffer("mean", torch.tensor(0.4161))
        self.register_buffer("std", torch.tensor(0.1688))

    def forward(self, x):
        B, T, W, H = x.shape
        # Normalize and Reshape
        x = (x.view(B * T, 1, W, H) / 255.0 - self.mean) / self.std

        # Forward Pass
        x = self.frontend(x)
        x = x.view(B, T, -1).transpose(1, 2)
        x = self.tcn(x)
        x = self.conv1d(x)
        return x.transpose(1, 2)


class ResNetLayer(nn.Module):
    """
    A ResNet layer used to build the ResNet network.
    Architecture:
    --> conv-bn-relu -> conv -> + -> bn-relu -> conv-bn-relu -> conv -> + -> bn-relu -->
     |                        |   |                                    |
     -----> downsample ------>    ------------------------------------->
    """

    def __init__(self, inplanes, outplanes, stride):
        super(ResNetLayer, self).__init__()
        self.conv1a = nn.Conv2d(
            inplanes, outplanes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1a = nn.BatchNorm2d(outplanes, momentum=0.01, eps=0.001)
        self.conv2a = nn.Conv2d(
            outplanes, outplanes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.stride = stride
        self.downsample = nn.Conv2d(
            inplanes, outplanes, kernel_size=(1, 1), stride=stride, bias=False
        )
        self.outbna = nn.BatchNorm2d(outplanes, momentum=0.01, eps=0.001)

        self.conv1b = nn.Conv2d(
            outplanes, outplanes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1b = nn.BatchNorm2d(outplanes, momentum=0.01, eps=0.001)
        self.conv2b = nn.Conv2d(
            outplanes, outplanes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.outbnb = nn.BatchNorm2d(outplanes, momentum=0.01, eps=0.001)
        return

    def forward(self, inputBatch):
        batch = F.relu(self.bn1a(self.conv1a(inputBatch)))
        batch = self.conv2a(batch)
        if self.stride == 1:
            residualBatch = inputBatch
        else:
            residualBatch = self.downsample(inputBatch)
        batch = batch + residualBatch
        intermediateBatch = batch
        batch = F.relu(self.outbna(batch))

        batch = F.relu(self.bn1b(self.conv1b(batch)))
        batch = self.conv2b(batch)
        residualBatch = intermediateBatch
        batch = batch + residualBatch
        outputBatch = F.relu(self.outbnb(batch))
        return outputBatch


class ResNet(nn.Module):
    """
    An 18-layer ResNet architecture.
    """

    def __init__(self):
        super(ResNet, self).__init__()
        self.layer1 = ResNetLayer(64, 64, stride=1)
        self.layer2 = ResNetLayer(64, 128, stride=2)
        self.layer3 = ResNetLayer(128, 256, stride=2)
        self.layer4 = ResNetLayer(256, 512, stride=2)
        self.avgpool = nn.AvgPool2d(kernel_size=(4, 4), stride=(1, 1))

        return

    def forward(self, inputBatch):
        batch = self.layer1(inputBatch)
        batch = self.layer2(batch)
        batch = self.layer3(batch)
        batch = self.layer4(batch)
        outputBatch = self.avgpool(batch)
        return outputBatch


class CausalLayerNorm(nn.Module):
    def __init__(self, channel_size, eps=1e-8):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1, channel_size, 1))
        self.beta = nn.Parameter(torch.zeros(1, channel_size, 1))
        self.eps = eps

    def forward(self, y):
        mean = y.mean(dim=1, keepdim=True)
        var = ((y - mean) ** 2).mean(dim=1, keepdim=True)
        y_norm = (y - mean) / torch.sqrt(var + self.eps)
        return self.gamma * y_norm + self.beta


class visualFrontend(nn.Module):
    """
    A visual feature extraction module. Generates a 512-dim feature vector per video frame.
    Architecture: A 2D convolution block followed by an 18-layer ResNet.
    """

    def __init__(self):
        super(visualFrontend, self).__init__()
        self.frontend2D = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64, momentum=0.01, eps=0.001),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.resnet = ResNet()
        return

    def forward(self, inputBatch):
        batchsize = inputBatch.shape[0]
        batch = self.frontend2D(inputBatch)

        outputBatch = self.resnet(batch)
        outputBatch = outputBatch.reshape(batchsize, -1, 512)
        outputBatch = outputBatch.transpose(1, 2)
        outputBatch = outputBatch.transpose(1, 2).transpose(0, 1)
        return outputBatch


# With Causal Convolution 1D
class DSConv1d(nn.Module):
    def __init__(self, dilation):
        super(DSConv1d, self).__init__()
        self.net = nn.Sequential(
            nn.ReLU(),
            CausalLayerNorm(512),
            CConv1d(512, 512, 3, stride=1, dilation=dilation, groups=512, bias=False),
            nn.PReLU(),
            CausalLayerNorm(512),
            CConv1d(512, 512, 1, bias=False),
        )

    def forward(self, x):
        out = self.net(x)
        return out + x


class visualTCN(nn.Module):
    def __init__(self):
        super(visualTCN, self).__init__()
        stacks = []
        for i in range(5):
            stacks += [DSConv1d(dilation=2**i)]
        self.net = nn.Sequential(*stacks)  # Visual Temporal Network V-TCN

    def forward(self, x):
        out = self.net(x)
        return out


class visualConv1D(nn.Module):
    def __init__(self):
        super(visualConv1D, self).__init__()
        self.net = nn.Sequential(
            CConv1d(512, 256, 5, stride=1),
            CausalLayerNorm(256),
            nn.ReLU(),
            CConv1d(256, 256, 1),
        )

    def forward(self, x):
        out = self.net(x)
        return out

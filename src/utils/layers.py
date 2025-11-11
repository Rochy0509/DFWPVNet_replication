import torch.nn as nn
from torch.nn import Sequential

def Conv3x3(in_channel, out_channel, stride = 1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_channel, out_channel, 3, stride=stride, padding=1, bias=False)

def MakeLayer(in_channel, out_channel, blocks, stride):
    """Create sequence of blocks"""
    layers = [BasicBlock(in_channel, out_channel, stride)]
    for _ in range(blocks - 1):
        layers.append(BasicBlock(out_channel, out_channel, 1))
    return Sequential(*layers)

class BasicBlock(nn.Module):
    """Basic block for ResNet"""

    def __init__(self, in_channel, out_channel, stride = 1):
        super().__init__()

        #first convolutional layer
        self.conv1 = Conv3x3(in_channel, out_channel, stride)
        #first Batch norm
        self.bn1 = nn.BatchNorm2d(out_channel)
        #Rectified linear activiation unit (ReLU)
        self.relu = nn.ReLU(inplace=True)
        #second convolutional layer
        self.conv2 = Conv3x3(out_channel, out_channel, stride)
        #second batch norm
        self.bn2 = nn.BatchNorm2d(out_channel)

        #downsample
        self.downsample = (nn.Sequential(nn.Conv2d(in_channel, out_channel, stride, bias=False),
                                         nn.BatchNorm2d(out_channel))
                                         if (stride != 1 or in_channel != out_channel) else nn.Identity())
        
    def forward(self, x):
        """Forward pass of the basic block"""
        y = self.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        x = self.downsample(x)
        return self.relu(x+y)
    
class Stem(nn.Module):
    """Stem block for the encoder"""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)

    def forward(self, x):
        """Forward pass of the stem block"""
        return self.pool(self.relu(self.bn(self.conv(x))))
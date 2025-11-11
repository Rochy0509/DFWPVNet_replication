import torch.nn as nn
from .utils.layers import Stem, MakeLayer

class ResNet18Backbone(nn.Module):
    """ResNet-18 encoder for feature extraction"""

    def __init__(self):
        super().__init__()
        self.stem = Stem()
        #Stride in cnn is the # of pixels by which the kernel moves across the input image
        self.layer1 = MakeLayer(64, 64, blocks=2, stride=1) 
        self.layer2 = MakeLayer(64, 128, blocks=2, stride=2)
        self.layer3 = MakeLayer(128, 256, blocks=2, stride=2)
        self.layer4 = MakeLayer(256, 512, blocks=2, stride=2)

    def forward(self, x):
        """Forward pass of the encoder"""

        s = self.stem(x) # H/4
        f1 = self.layer1(s) # H/4
        f2 = self.layer2(f1) # H/8
        f3 = self.layer3(f2)# H/16
        f4 = self.layer4(f3) # H/32
        return s, f1, f2, f3, f4

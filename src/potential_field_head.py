import torch.nn as nn

class PotentialFieldHead(nn.Module):
    """Predicts potential weights based on data field theory"""

    def __init__(self, in_channels=32, num_keypoints=9, use_sigmoid=True):
        super().__init__()
        self.K = num_keypoints
        self.use_sigmoid = use_sigmoid
        self.conv = nn.Conv2d(in_channels, num_keypoints, kernel_size=1, bias=True)

        if use_sigmoid:
            self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.conv(x)
        if self.use_sigmoid:
            x = self.sigmoid(x)
        return x
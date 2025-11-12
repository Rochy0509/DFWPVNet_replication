import torch.nn as nn

class VectorFieldHead(nn.Module):
    """Predicts unit vectors from each pixel to K keypoints"""

    def __init__(self, in_channels=32, num_keypoints=9):
        super().__init__()
        self.K = num_keypoints
        self.conv = nn.Conv2d(in_channels, 2 * num_keypoints, kernel_size=1, bias=True)
    
    def forward(self, x):
        return self.conv(x)
    
    
import torch.nn as nn

class MaskHead(nn.Module):
    """
    Binary segmentation head for object mask prediction.
    """
    def __init__(self, in_channels=32, num_classes=2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, num_classes, 
                              kernel_size=1, bias=True)
    
    def forward(self, x):
        """
        Returns:
            Mask logits [B, 2, H, W]
        """
        return self.conv(x)
import torch.nn as nn
import torch
from .utils.layers import Conv3x3

class UpsampleBlock(nn.Module):
  """
  Upsampling block for the decoder.
  """
  def __init__(self, in_channel, skip_channel, out_channel ):
    """
    Initializes the UpsampleBlock.
    """
    super().__init__()
    self.conv = nn.Sequential(Conv3x3(in_channel, out_channel),
                              nn.BatchNorm2d(out_channel), nn.ReLU(inplace=True))
    self.fuse = nn.Sequential(Conv3x3(out_channel+skip_channel, out_channel),
                              nn.BatchNorm2d(out_channel), nn.ReLU(inplace=True))
    self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

  def forward(self, x, skip):
    """
    Forward pass of the UpsampleBlock.
    """
    x = self.conv(x)
    x = self.upsample(x)
    x = torch.cat([x, skip], dim=1)
    return self.fuse(x)
  
class UNetDecoder(nn.Module):
    """
    U-Net style decoder with skip connections.
    """
    def __init__(self):
        super().__init__()
        # Decoder blocks
        self.up3 = UpsampleBlock(512, 256, 256)  # D3
        self.up2 = UpsampleBlock(256, 128, 128)  # D2
        self.up1 = UpsampleBlock(128, 64, 64)    # D1
        self.up0 = UpsampleBlock(64, 64, 64)     # D0 (fuse stem)
        
        # Final upsampling to original resolution
        self.final_upsample = nn.Upsample(scale_factor=2, 
                                          mode='bilinear',
                                          align_corners=False)
        
        # Compression layer (64+3 RGB → 32)
        self.compress = nn.Sequential(
            Conv3x3(64 + 3, 32),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, rgb_input, stem, f1, f2, f3, f4):
        """
        Returns:
            Fused features [B, 32, H, W] ready for prediction heads
        """
        # Progressive upsampling with skip connections
        x = self.up3(f4, f3)  # H/16
        x = self.up2(x, f2)   # H/8
        x = self.up1(x, f1)   # H/4
        x = self.up0(x, stem) # H/2
        
        # Final upsample to H,W
        x = self.final_upsample(x)
        
        # Fuse with RGB input
        x = torch.cat([x, rgb_input], dim=1)
        x = self.compress(x)
        
        return x  # [B, 32, H, W]
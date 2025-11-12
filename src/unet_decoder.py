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
        self.up3 = UpsampleBlock(512, 256, 256)  # D3: H/32 → H/16
        self.up2 = UpsampleBlock(256, 128, 128)  # D2: H/16 → H/8
        self.up1 = UpsampleBlock(128, 64, 64)    # D1: H/8 → H/4

        # Additional skip connection with stem (both at H/4)
        self.stem_fuse = nn.Sequential(
            Conv3x3(64 + 64, 64),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Final upsampling to original resolution (H/4 → H)
        self.final_upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # H/4 → H/2
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)   # H/2 → H
        )

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
        x = self.up3(f4, f3)  # H/32 → H/16, merge with f3
        x = self.up2(x, f2)   # H/16 → H/8, merge with f2
        x = self.up1(x, f1)   # H/8 → H/4, merge with f1

        # Fuse with stem at H/4 resolution
        x = torch.cat([x, stem], dim=1)
        x = self.stem_fuse(x)  # Still H/4

        # Final upsample to H,W (4x upsampling)
        x = self.final_upsample(x)

        # Fuse with RGB input
        x = torch.cat([x, rgb_input], dim=1)
        x = self.compress(x)

        return x  # [B, 32, H, W]
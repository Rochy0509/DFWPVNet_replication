import torch
import torch.nn as nn
from backbone import ResNet18Backbone
from unet_decoder import UNetDecoder
from mask_head import MaskHead
from vector_field_head import VectorFieldHead  
from potential_field_head import PotentialFieldHead 

class DFWPVNet(nn.Module):
    """
    Data Field Weighting based Pixel-wise Voting Network.
    """
    def __init__(self, num_keypoints=9):
        super().__init__()
        self.K = num_keypoints
        
        # Backbone: Feature extraction
        self.backbone = ResNet18Backbone()
        
        # Neck: Feature fusion
        self.neck = UNetDecoder()
        
        # Heads: Task-specific predictions
        self.mask_head = MaskHead(in_channels=32, num_classes=2)
        self.vector_field_head = VectorFieldHead(in_channels=32, num_keypoints=num_keypoints)
        self.potential_field_head = PotentialFieldHead(in_channels=32, num_keypoints=num_keypoints)
    
    def forward(self, x):
        
        # Extract features
        stem, f1, f2, f3, f4 = self.backbone(x)
        
        # Fuse and upsample
        features = self.neck(x, stem, f1, f2, f3, f4)
        
        # Parallel predictions
        mask = self.mask_head(features)
        vertex = self.vector_field_head(features)
        potential = self.potential_field_head(features)
        
        return mask, vertex, potential
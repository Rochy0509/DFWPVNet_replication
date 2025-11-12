import torch.nn as nn 
import torch.nn.functional as F

class DFWPVNetLoss(nn.Module):
    """
    Loss function based on the DFW-PVNet paper

    L_total = lambda1*L_mask + lambda2 * L_vertex + lambda3 * L_potential

    Paper values:
        lambda1 = 1.0 (mask)
        lambda2 = 0.1 (vertex)
        lambda3 = 10.0 (potential)    
    """
    def __init__(self, lambda1=1.0, lambda2=0.1, lambda3=10.0):
        super().__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3

        #Mask loss: Cross-entropy
        self.mask_loss = nn.CrossEntropyLoss()

        #Vertex & Potential: smoothh L1
        self.smooth_l1 = nn.SmoothL1Loss(reduction='mean')
    
    def forward(self, pred_mask, pred_vertex, pred_potential, 
                gt_mask, gt_vertex, gt_potential):
        """
        Args:
            pred_mask: [B, 2, H, W]
            pred_vertex: [B, 2*K, H, W]
            pred_potential: [B, K, H, W]
            gt_mask: [B, H, W] (class indices)
            gt_vertex: [B, 2*K, H, W]
            gt_potential: [B, K, H, W]
        """
        #Mask loss (only on object pixels)
        loss_mask = self.mask_loss(pred_mask, gt_mask)

        #Vertex loss  (only on object pixels)
        mask_obj = (gt_mask > 0).unsqueeze(1).float()
        loss_vertex = self.smooth_l1(
            pred_vertex * mask_obj,
            gt_vertex * mask_obj
        )

        #Potential loss (only on object pixels)
        loss_potential = self.smooth_l1(
            pred_potential * mask_obj, 
            gt_potential * mask_obj
        )

        #Total loss
        total_loss = (self.lambda1 * loss_mask +
                      self.lambda2 * loss_vertex +
                      self.lambda3 * loss_potential)
        
        return {
            'total': total_loss,
            'mask': loss_mask,
            'vertex': loss_vertex,
            'potential': loss_potential
        }

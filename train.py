# train.py (at project root, not in src/)

import torch
from torch.utils.data import DataLoader
from src.dfwpvnet import DFWPVNet
from src.loss_functions import DFWPVNetLoss
from src.linemod_dataset import LINEMODDataset

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    
    for batch_idx, batch in enumerate(dataloader):
        images = batch['image'].to(device)
        gt_mask = batch['mask'].to(device)
        gt_vertex = batch['vertex'].to(device)
        gt_potential = batch['potential'].to(device)
        
        # Forward
        pred_mask, pred_vertex, pred_potential = model(images)
        
        # Loss
        losses = criterion(pred_mask, pred_vertex, pred_potential,
                          gt_mask, gt_vertex, gt_potential)
        
        # Backward
        optimizer.zero_grad()
        losses['total'].backward()
        optimizer.step()
        
        total_loss += losses['total'].item()
        
        if batch_idx % 10 == 0:
            print(f"  Batch {batch_idx}/{len(dataloader)}: Loss = {losses['total'].item():.4f}")
    
    return total_loss / len(dataloader)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Model
    model = DFWPVNet(num_keypoints=9).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss
    criterion = DFWPVNetLoss(lambda1=1.0, lambda2=0.1, lambda3=10.0)
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    # Data
    data_dir = 'path/to/LINEMOD'  # UPDATE THIS
    train_dataset = LINEMODDataset(data_dir, split='train')
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
    
    # Training loop
    for epoch in range(200):
        print(f"\nEpoch {epoch+1}/200")
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        scheduler.step()
        print(f"Epoch {epoch+1}: Average Loss = {loss:.4f}, LR = {scheduler.get_last_lr()[0]:.6f}")
        
        # Save checkpoint
        if (epoch + 1) % 20 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, f'checkpoint_epoch_{epoch+1}.pth')
            print(f"Saved checkpoint at epoch {epoch+1}")

if __name__ == '__main__':
    main()
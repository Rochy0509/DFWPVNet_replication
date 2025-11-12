import torch
from torch.utils.data import DataLoader
from dfwpvnet import DFWPVNet
from loss_functions import DFWPVNetLoss
from linemod_dataset import LINEMODDataset

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    
    for batch in dataloader:
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
    
    return total_loss / len(dataloader)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Model
    model = DFWPVNet(num_keypoints=9).to(device)
    
    # Loss
    criterion = DFWPVNetLoss(lambda1=1.0, lambda2=0.1, lambda3=10.0)
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    # Data
    train_dataset = LINEMODDataset('path/to/data', split='train')
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    
    # Training loop
    for epoch in range(200):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        scheduler.step()
        print(f"Epoch {epoch}: Loss = {loss:.4f}")

if __name__ == '__main__':
    main()
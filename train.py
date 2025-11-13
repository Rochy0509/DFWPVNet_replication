import torch
from torch.utils.data import DataLoader
from src.dfwpvnet import DFWPVNet
from src.loss_functions import DFWPVNetLoss
from src.linemod_dataset import LINEMODDataset
import matplotlib.pyplot as plt
import matplotlib
# Try interactive backend, fallback to Agg if not available
try:
    matplotlib.use('TkAgg')
except:
    try:
        matplotlib.use('Qt5Agg')
    except:
        matplotlib.use('Agg')
        print("Warning: Interactive display not available, plots will be saved to files only")
plt.ion()  # Enable interactive mode

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

def plot_losses(epoch_losses, fig=None, save_path='training_loss.png'):
    """
    Dynamically plot training losses, display interactively, and save to file.
    """
    if fig is None:
        fig = plt.figure(figsize=(10, 6))

    plt.clf()  # Clear the figure
    epochs = range(1, len(epoch_losses) + 1)
    plt.plot(epochs, epoch_losses, 'b-', linewidth=2, marker='o', markersize=4)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Average Loss', fontsize=12)
    plt.title('Training Loss Over Time', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save to file
    plt.savefig(save_path, dpi=150, bbox_inches='tight')

    # Show interactively
    plt.draw()
    plt.pause(0.001)  # Brief pause to update the plot

    print(f"  Loss plot updated and saved to {save_path}")
    return fig

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
    data_dir = 'dataset/linemod/linemod'
    train_dataset = LINEMODDataset(data_dir, split='train')
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)

    # Loss tracking for visualization
    epoch_losses = []
    loss_fig = None  # Will hold the figure for dynamic plotting

    # Training loop
    for epoch in range(200):
        print(f"\nEpoch {epoch+1}/200")
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        scheduler.step()
        print(f"Epoch {epoch+1}: Average Loss = {loss:.4f}, LR = {scheduler.get_last_lr()[0]:.6f}")

        # Track and visualize loss
        epoch_losses.append(loss)

        # Update plot every 5 epochs (or every epoch for first 10 epochs)
        if (epoch + 1) <= 10 or (epoch + 1) % 5 == 0:
            loss_fig = plot_losses(epoch_losses, fig=loss_fig)

        # Save checkpoint
        if (epoch + 1) % 20 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, f'checkpoint_epoch_{epoch+1}.pth')
            print(f"Saved checkpoint at epoch {epoch+1}")

    # Final plot save
    print("\n" + "="*50)
    print("Training completed!")
    plot_losses(epoch_losses, fig=loss_fig, save_path='final_training_loss.png')
    print(f"Final average loss: {epoch_losses[-1]:.4f}")
    print("="*50)

    # Keep plot window open
    plt.ioff()  # Turn off interactive mode
    plt.show()  # Keep window open until manually closed

if __name__ == '__main__':
    main()
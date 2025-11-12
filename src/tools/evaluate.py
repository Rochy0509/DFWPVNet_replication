import torch
from torch.utils.data import DataLoader
from src.dfwpvnet import DFWPVNet
from src.linemod_dataset import LINEMODDataset
from src.inference.keypoint_localizer import KeypointLocalizer
from src.inference.pose_estimator import PoseEstimator
from src.evaluation.metrics import MetricsCalculator

def evaluate(model, dataloader, model_points_dict, diameters_dict, device='cuda'):
    """
    Evaluate model on dataset.
    """
    model.eval()
    localizer = KeypointLocalizer(num_keypoints=9, num_samples=512, tau=0.999)
    pose_estimator = PoseEstimator()
    
    # Symmetric objects in LINEMOD (eggbox, glue)
    symmetric_objects = [8, 9]  
    
    metrics_calc = MetricsCalculator(
        model_points_dict, 
        diameters_dict,
        symmetric_objects
    )
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            gt_pose_R = batch['pose_R']
            gt_pose_T = batch['pose_T']
            keypoints_3d = batch['keypoints_3d']
            camera_K = batch['camera_K']
            class_ids = batch['class_id']
            
            # Forward pass
            pred_mask, pred_vertex, pred_potential = model(images)
            
            # Process each sample in batch
            for i in range(len(images)):
                # Get predictions
                mask = torch.argmax(pred_mask[i], dim=0)
                vertex = pred_vertex[i]
                potential = pred_potential[i]
                
                # Localize keypoints
                keypoints_2d = localizer.localize(mask, vertex, potential)
                
                # Estimate pose
                success, pred_R, pred_T = pose_estimator.estimate_pose(
                    keypoints_2d,
                    keypoints_3d[i].cpu().numpy(),
                    camera_K[i].cpu().numpy(),
                    use_ransac=True
                )
                
                if not success:
                    continue
                
                # Update metrics
                metrics_calc.update(
                    pred_R, pred_T,
                    gt_pose_R[i].cpu().numpy(),
                    gt_pose_T[i].cpu().numpy(),
                    class_ids[i].item(),
                    camera_K[i].cpu().numpy()
                )
    
    # Compute final metrics
    results = metrics_calc.compute()
    return results


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = DFWPVNet(num_keypoints=9).to(device)
    checkpoint = torch.load('checkpoint.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load dataset
    test_dataset = LINEMODDataset('path/to/LINEMOD', split='test')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # Prepare model points and diameters (load from dataset)
    model_points_dict = {}  # TODO: Load from dataset
    diameters_dict = {}  # TODO: Load from dataset
    
    # Evaluate
    results = evaluate(model, test_loader, model_points_dict, diameters_dict, device)
    
    print("\n" + "="*50)
    print("Evaluation Results:")
    print("="*50)
    for metric_name, value in results.items():
        print(f"{metric_name}: {value:.2f}%")

if __name__ == '__main__':
    main()

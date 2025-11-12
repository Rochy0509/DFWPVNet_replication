import torch
import cv2
import numpy as np
from src.dfwpvnet import DFWPVNet
from inference.keypoint_localizer import KeypointLocalizer
from inference.pose_estimator import PoseEstimator

def inference_single_image(model, image_path, keypoints_3d, camera_K, 
                           device='cuda'):
    """
    Run inference on a single image.
    """
    # Load image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Preprocess
    image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
    image_tensor = image_tensor.unsqueeze(0).to(device)  # [1, 3, H, W]
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        pred_mask, pred_vertex, pred_potential = model(image_tensor)
    
    # Get predictions (remove batch dimension)
    mask = torch.argmax(pred_mask[0], dim=0)  # [H, W]
    vertex = pred_vertex[0]  # [2*K, H, W]
    potential = pred_potential[0]  # [K, H, W]
    
    # Localize keypoints
    localizer = KeypointLocalizer(num_keypoints=9, num_samples=512, tau=0.999)
    keypoints_2d = localizer.localize(mask, vertex, potential)
    
    # Estimate pose
    pose_estimator = PoseEstimator()
    success, R, T = pose_estimator.estimate_pose(
        keypoints_2d, keypoints_3d, camera_K, use_ransac=True
    )
    
    if not success:
        print("Pose estimation failed!")
        return None, None, keypoints_2d
    
    return R, T, keypoints_2d


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = DFWPVNet(num_keypoints=9).to(device)
    checkpoint = torch.load('checkpoint.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Example usage
    image_path = 'path/to/image.png'
    keypoints_3d = np.random.randn(9, 3)  # Load from dataset
    camera_K = np.eye(3)  # Load camera intrinsics
    
    R, T, keypoints_2d = inference_single_image(
        model, image_path, keypoints_3d, camera_K, device
    )
    
    print(f"Estimated pose:")
    print(f"Rotation:\n{R}")
    print(f"Translation:\n{T}")
    print(f"2D Keypoints:\n{keypoints_2d}")

if __name__ == '__main__':
    main()
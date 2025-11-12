import numpy as np
import torch

def generate_potential_field(keypoints_2d, mask, sigma=0.06, k=2, m=1):
    """Generate potential field ground truth using data field theory"""

    H, W = mask.shape
    K = len(keypoints_2d)
    potential = np.zeros((K, H, W), dtype=np.float32)

    y_coords, x_coords = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')

    for i, (kx, ky) in enumerate(keypoints_2d):
        #Euclidean distance 
        dist = np.sqrt((x_coords - kx)**2 + (y_coords - ky)**2)

        #Data field formula 
        potential[i] = m * np.exp(-((dist / sigma)**k))

        #Apply mask
        potential[i] *= mask

    return torch.from_numpy(potential)

def generate_vector_field(keypoints_2d, mask):
    """Generate unit vector to keypoints"""

    H, W = mask.shape
    K = len(keypoints_2d)
    vectors = np.zeros((2*K, H, W), dtype=np.float32)

    y_coords, x_coords = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')

    for i, (kx, ky) in enumerate(keypoints_2d):
        #Vector from pixel to keypoint
        vx = kx - x_coords
        vy = ky - y_coords

        #Normalize
        norm = np.sqrt(vx**2 + vy**2) + 1e-8
        vx = vx / norm
        vy = vy / norm

        #Apply mask and store
        vectors[2*i] = vx * mask
        vectors[2*i + 1] = vy * mask
    
    return torch.from_numpy(vectors)

import numpy as np  
import trimesh  

def farthest_point_sampling(points, num_samples):
    """
    Farthest Point Sampling algorithm.
    """
    N = points.shape[0]
    
    # Initialize
    sampled_indices = np.zeros(num_samples, dtype=np.int32)
    distances = np.full(N, np.inf)
    
    # Start with a random point (or the centroid)
    current_idx = np.random.randint(0, N)
    
    for i in range(num_samples):
        sampled_indices[i] = current_idx
        current_point = points[current_idx]
        
        # Update distances to the nearest sampled point
        dist_to_current = np.linalg.norm(points - current_point, axis=1)
        distances = np.minimum(distances, dist_to_current)
        
        # Select the farthest point as next sample
        current_idx = np.argmax(distances)
    
    sampled_points = points[sampled_indices]
    return sampled_points, sampled_indices

def load_3d_model(model_path):
    """Load 3D model using trimesh only"""
    try:
        mesh = trimesh.load(model_path, force='mesh')
        points = np.array(mesh.vertices)
        print(f"Loaded {len(points)} vertices from {model_path}")
        return points
    except Exception as e:
        raise RuntimeError(f"Failed to load 3D model: {e}")

def generate_keypoints_from_model(model_path, num_keypoints=8, include_center=True):
    """
    Generate 3D keypoints using FPS.
    """
    # Load 3D model points
    points = load_3d_model(model_path)
    
    # Sample keypoints using FPS
    keypoints, _ = farthest_point_sampling(points, num_keypoints)
    
    if include_center:
        # Calculate object center
        center = np.mean(points, axis=0, keepdims=True)  # [1, 3]
        # Add center as 9th keypoint
        keypoints = np.vstack([keypoints, center])  # [9, 3]
    
    return keypoints
import numpy as np
import open3d as o3d  
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
    """
    Load 3D model from PLY or OBJ file.
    """
    if model_path.endswith('.ply'):
        # Option 1: Using Open3D
        mesh = o3d.io.read_triangle_mesh(model_path)
        points = np.asarray(mesh.vertices)
        
        # Option 2: Using trimesh (alternative)
        # mesh = trimesh.load(model_path)
        # points = mesh.vertices
        
    elif model_path.endswith('.obj'):
        mesh = trimesh.load(model_path)
        points = mesh.vertices
    else:
        raise ValueError(f"Unsupported file format: {model_path}")
    
    return points

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
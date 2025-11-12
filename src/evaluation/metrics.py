import numpy as np

def compute_add_metric(pred_R, pred_T, gt_R, gt_T, model_points, diameter, 
                       use_add_s=False):
    """
    ADD or ADD-S metric: Average Distance of Model Points.
    """
    # Ensure correct shapes
    if pred_T.ndim == 1:
        pred_T = pred_T.reshape(3, 1)
    if gt_T.ndim == 1:
        gt_T = gt_T.reshape(3, 1)
    
    # Transform model points
    pred_pts = (pred_R @ model_points.T).T + pred_T.T  # [N, 3]
    gt_pts = (gt_R @ model_points.T).T + gt_T.T  # [N, 3]
    
    if use_add_s:
        # ADD-S: For each predicted point, find closest GT point
        distances = []
        for pred_pt in pred_pts:
            dist = np.linalg.norm(gt_pts - pred_pt, axis=1)
            distances.append(np.min(dist))
        avg_distance = np.mean(distances)
    else:
        # ADD: Direct point-to-point distance
        distances = np.linalg.norm(pred_pts - gt_pts, axis=1)
        avg_distance = np.mean(distances)
    
    # Threshold: 10% of diameter
    threshold = 0.1 * diameter
    accuracy = 1 if avg_distance < threshold else 0
    
    return accuracy, avg_distance


def compute_2d_projection_metric(pred_R, pred_T, gt_R, gt_T, 
                                  model_points, camera_K, threshold=5.0):
    """
    2D projection metric: Average pixel error < threshold.
    """
    # Ensure correct shapes
    if pred_T.ndim == 1:
        pred_T = pred_T.reshape(3, 1)
    if gt_T.ndim == 1:
        gt_T = gt_T.reshape(3, 1)
    
    # Transform to camera coordinates
    pred_cam = (pred_R @ model_points.T) + pred_T  # [3, N]
    gt_cam = (gt_R @ model_points.T) + gt_T  # [3, N]
    
    # Project to image plane
    pred_2d = camera_K @ pred_cam  # [3, N]
    gt_2d = camera_K @ gt_cam  # [3, N]
    
    # Convert to pixel coordinates
    pred_pixels = pred_2d[:2, :] / pred_2d[2:3, :]  # [2, N]
    gt_pixels = gt_2d[:2, :] / gt_2d[2:3, :]  # [2, N]
    
    # Calculate pixel error
    pixel_distances = np.linalg.norm(pred_pixels - gt_pixels, axis=0)
    avg_pixel_error = np.mean(pixel_distances)
    
    accuracy = 1 if avg_pixel_error < threshold else 0
    
    return accuracy, avg_pixel_error


def compute_5cm5deg_metric(pred_R, pred_T, gt_R, gt_T):
    """
    5cm x 5° metric: Both rotation and translation within thresholds.
    """
    # Ensure correct shapes
    if pred_T.ndim == 1:
        pred_T = pred_T.reshape(3, 1)
    if gt_T.ndim == 1:
        gt_T = gt_T.reshape(3, 1)
    
    # Translation error (convert to cm)
    trans_diff = pred_T - gt_T
    trans_error = np.linalg.norm(trans_diff) * 100  # meters to cm
    
    # Rotation error
    R_diff = pred_R @ gt_R.T
    trace = np.trace(R_diff)
    # Clamp to valid range for arccos
    trace = np.clip((trace - 1) / 2, -1.0, 1.0)
    rot_error = np.arccos(trace) * 180 / np.pi  # radians to degrees
    
    # Check thresholds
    trans_ok = trans_error < 5.0  # < 5 cm
    rot_ok = rot_error < 5.0  # < 5 degrees
    
    accuracy = 1 if (trans_ok and rot_ok) else 0
    
    return accuracy, rot_error, trans_error


class MetricsCalculator:
    """Batch metrics calculation with running statistics."""
    
    def __init__(self, model_points_dict, diameters_dict, 
                 symmetric_objects=None):

        self.model_points_dict = model_points_dict
        self.diameters_dict = diameters_dict
        self.symmetric_objects = symmetric_objects or []
        
        self.reset()
    
    def reset(self):
        """Reset all counters."""
        self.add_scores = []
        self.proj_scores = []
        self.cm5deg_scores = []
        self.num_samples = 0
    
    def update(self, pred_R, pred_T, gt_R, gt_T, class_id, camera_K):
        """
        Update metrics with a single prediction.
        """
        # Get model info
        model_points = self.model_points_dict[class_id]
        diameter = self.diameters_dict[class_id]
        use_add_s = class_id in self.symmetric_objects
        
        # Compute ADD(-S)
        add_acc, _ = compute_add_metric(
            pred_R, pred_T, gt_R, gt_T, 
            model_points, diameter, use_add_s
        )
        self.add_scores.append(add_acc)
        
        # Compute 2D projection
        proj_acc, _ = compute_2d_projection_metric(
            pred_R, pred_T, gt_R, gt_T,
            model_points, camera_K
        )
        self.proj_scores.append(proj_acc)
        
        # Compute 5cm×5°
        cm5deg_acc, _, _ = compute_5cm5deg_metric(
            pred_R, pred_T, gt_R, gt_T
        )
        self.cm5deg_scores.append(cm5deg_acc)
        
        self.num_samples += 1
    
    def compute(self):
        """
        Compute final metrics.
        """
        if self.num_samples == 0:
            return {
                'ADD(-S)': 0.0,
                '2D-Proj': 0.0,
                '5cm5deg': 0.0
            }
        
        return {
            'ADD(-S)': np.mean(self.add_scores) * 100,  # Percentage
            '2D-Proj': np.mean(self.proj_scores) * 100,
            '5cm5deg': np.mean(self.cm5deg_scores) * 100
        }
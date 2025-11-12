import numpy as np
import cv2

class PoseEstimator:
    """Estimate 6D pose from 2D-3D correspondences using PnP solver."""
    
    def __init__(self, method=cv2.SOLVEPNP_ITERATIVE):
        self.method = method
    
    def estimate_pose(self, keypoints_2d, keypoints_3d, camera_K, 
                     use_ransac=False):
        """
        Estimate 6D pose from 2D-3D correspondences.
        """
        # Ensure correct types
        keypoints_2d = np.ascontiguousarray(keypoints_2d, dtype=np.float32)
        keypoints_3d = np.ascontiguousarray(keypoints_3d, dtype=np.float32)
        camera_K = np.ascontiguousarray(camera_K, dtype=np.float32)
        
        # No distortion
        dist_coeffs = np.zeros(4, dtype=np.float32)
        
        try:
            if use_ransac:
                # Use RANSAC for robustness
                success, rvec, tvec, inliers = cv2.solvePnPRansac(
                    keypoints_3d,
                    keypoints_2d,
                    camera_K,
                    dist_coeffs,
                    flags=self.method
                )
            else:
                # Standard PnP
                success, rvec, tvec = cv2.solvePnP(
                    keypoints_3d,
                    keypoints_2d,
                    camera_K,
                    dist_coeffs,
                    flags=self.method
                )
            
            if not success:
                return False, None, None
            
            # Convert rotation vector to matrix
            R, _ = cv2.Rodrigues(rvec)
            
            return True, R, tvec
            
        except cv2.error as e:
            print(f"PnP solver failed: {e}")
            return False, None, None
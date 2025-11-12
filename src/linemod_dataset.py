import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import os
import json
import yaml
from .ground_truth_generator import generate_potential_field, generate_vector_field
from .keypoint_selector import generate_keypoints_from_model

class LINEMODDataset(Dataset):
    
    def __init__(self, data_dir, split='train', num_keypoints=9, transform=None):
        self.data_dir = data_dir
        self.split = split
        self.K = num_keypoints
        self.transform = transform

        # Load 3D keypoints for each class
        self.keypoints_3d_dict = self._load_3d_keypoints()

        # Load annotations
        self.annotations = self._load_annotations()
        
        print(f"Loaded {len(self.annotations)} samples for {split} split")

    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        ann = self.annotations[idx]

        #Load image
        image = cv2.imread(ann['image_path'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        #Load mask
        mask = cv2.imread(ann['mask_path'], cv2.IMREAD_GRAYSCALE)
        mask = (mask > 0).astype(np.float32)

        #Project 3D keypoints to 2D
        keypoints_2d = self._project_keypoints(
            ann['pose_R'], ann['pose_T'], ann['camera_K'], ann['class_id']
        )

        # Generate ground truth
        gt_vertex = generate_vector_field(keypoints_2d, mask)
        gt_potential = generate_potential_field(keypoints_2d, mask)

        # Convert to tensors
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        mask = torch.from_numpy(mask).long()

        return {
            'image': image,
            'mask': mask,
            'vertex': gt_vertex,
            'potential': gt_potential,
            'keypoints_2d': torch.from_numpy(keypoints_2d).float(),
            'class_id': ann['class_id']
        }
    
    def _project_keypoints(self, R, T, K, class_id):

        #Get 3D keypoints for this object class
        keypoints_3d = self.keypoints_3d_dict[class_id]

        #Ensure T is [3, 1]
        if T.ndim == 1:
            T = T.reshape(3, 1)
        
        # Transform 3D points to camera coordinate system
        # P_cam = R @ P_obj + T
        keypoints_cam = (R @ keypoints_3d.T) + T

        # Project to image plane using camera intrinsics
        # p = K @ P_cam
        keypoints_2d_homogeneous = K @ keypoints_cam

        # Convert from homogeneous to pixel coordinates
        # [x, y, z] -> [x/z, y/z]
        keypoints_2d = keypoints_2d_homogeneous[:2, :] / keypoints_2d_homogeneous[2:3, :]

        # Transpose to [num_keypoints, 2]
        keypoints_2d = keypoints_2d.T
        return keypoints_2d
    
    def _load_3d_keypoints(self):
        """
        Load or generate 3D keypoints for each object class.
        """
        keypoints_3d_dict = {}
        
        # LINEMOD has 13 classes
        class_names = [
            'ape', 'benchvise', 'cam', 'can', 'cat', 'driller', 'duck',
            'eggbox', 'glue', 'holepuncher', 'iron', 'lamp', 'phone'
        ]
        
        # Path to 3D models directory
        models_dir = os.path.join(self.data_dir, 'models')
        
        # Check if precomputed keypoints exist
        keypoints_cache_path = os.path.join(self.data_dir, 'keypoints_3d.json')
        
        if os.path.exists(keypoints_cache_path):
            # Load precomputed keypoints
            print(f"Loading precomputed 3D keypoints from {keypoints_cache_path}")
            with open(keypoints_cache_path, 'r') as f:
                keypoints_data = json.load(f)
                for class_id, keypoints_list in keypoints_data.items():
                    keypoints_3d_dict[int(class_id)] = np.array(keypoints_list)
        else:
            # Generate keypoints using FPS
            print("Generating 3D keypoints using FPS...")
            for class_id, class_name in enumerate(class_names, start=1):
                model_path = os.path.join(models_dir, f'obj_{class_id:06d}.ply')
                
                if not os.path.exists(model_path):
                    print(f"Warning: Model not found at {model_path}")
                    continue
                
                # Generate 9 keypoints (8 FPS + 1 center)
                keypoints_3d = generate_keypoints_from_model(
                    model_path, 
                    num_keypoints=8,
                    include_center=True
                )
                
                keypoints_3d_dict[class_id] = keypoints_3d
                print(f"Generated keypoints for {class_name} (class {class_id})")
            
            # Save for future use
            keypoints_to_save = {
                str(k): v.tolist() for k, v in keypoints_3d_dict.items()
            }
            with open(keypoints_cache_path, 'w') as f:
                json.dump(keypoints_to_save, f)
            print(f"Saved keypoints to {keypoints_cache_path}")
        
        return keypoints_3d_dict
    
    def _load_annotations(self):
        """Load LINEMOD dataset annotations in BOP format."""
        annotations = []

        class_names = [
            'ape', 'benchvise', 'cam', 'can', 'cat', 'driller', 'duck',
            'eggbox', 'glue', 'holepuncher', 'iron', 'lamp', 'phone'
        ]

        for class_id, class_name in enumerate(class_names, start=1):

            # Load split file to get frame IDs for this class
            split_file = os.path.join(self.data_dir, f'{class_id:06d}_{self.split}.txt')

            if not os.path.exists(split_file):
                print(f"Warning: Split file not found: {split_file}")
                continue

            with open(split_file, 'r') as f:
                split_lines = f.readlines()

            # BOP format: real/{class_id}/rgb/{frame_id}.png
            frame_ids = []
            for line in split_lines:
                line = line.strip()
                if line:
                    # Extract frame_id from path like "real/000001/rgb/000013.png"
                    parts = line.split('/')
                    if len(parts) >= 4:
                        frame_filename = parts[-1]  # "000013.png"
                        frame_id = int(frame_filename.split('.')[0])  # 13
                        frame_ids.append(frame_id)

            if not frame_ids:
                print(f"Warning: No frames found for class {class_id} in {self.split} split")
                continue

            # Load ground truth poses and camera info from BOP JSON files
            class_dir = os.path.join(self.data_dir, 'real', f'{class_id:06d}')

            if not os.path.exists(class_dir):
                print(f"Warning: Class directory not found: {class_dir}")
                continue

            gt_path = os.path.join(class_dir, 'scene_gt.json')
            camera_path = os.path.join(class_dir, 'scene_camera.json')

            if not os.path.exists(gt_path) or not os.path.exists(camera_path):
                print(f"Warning: scene_gt.json or scene_camera.json not found at {class_dir}")
                continue

            with open(gt_path, 'r') as f:
                gt_data = json.load(f)

            with open(camera_path, 'r') as f:
                camera_data = json.load(f)

            # Process each frame in the split
            rgb_dir = os.path.join(class_dir, 'rgb')
            mask_dir = os.path.join(class_dir, 'mask_visib')

            for frame_id in frame_ids:
                frame_key = str(frame_id)

                if frame_key not in gt_data or frame_key not in camera_data:
                    continue

                # Get pose for this frame (first object in list)
                pose_data = gt_data[frame_key][0]
                R = np.array(pose_data['cam_R_m2c']).reshape(3, 3)
                T = np.array(pose_data['cam_t_m2c']) / 1000.0  # Convert mm to meters

                # Get camera intrinsics
                K = np.array(camera_data[frame_key]['cam_K']).reshape(3, 3)

                # Construct paths
                image_path = os.path.join(rgb_dir, f'{frame_id:06d}.png')
                # Mask format: {frame_id:06d}_{instance_id:06d}.png
                mask_path = os.path.join(mask_dir, f'{frame_id:06d}_000000.png')

                # Check if files exist
                if not os.path.exists(image_path):
                    print(f"Warning: Image not found: {image_path}")
                    continue
                if not os.path.exists(mask_path):
                    print(f"Warning: Mask not found: {mask_path}")
                    continue

                annotations.append({
                    'image_path': image_path,
                    'mask_path': mask_path,
                    'pose_R': R,
                    'pose_T': T,
                    'camera_K': K,
                    'class_id': class_id,
                    'class_name': class_name,
                    'frame_id': frame_id
                })

        print(f"Loaded {len(annotations)} annotations from LINEMOD dataset")
        return annotations
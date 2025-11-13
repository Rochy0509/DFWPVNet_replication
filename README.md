# DFWPVNet Replication

This repository contains a PyTorch implementation of **DFW-PVNet** (Data Field Weighting based Pixel-wise Voting Network) for 6D object pose estimation. This is a course project for the AI & ML course at Ontario Tech University, where our team has replicated the DFW-PVNet method from the original research paper.

## Overview

DFW-PVNet is a deep learning approach for estimating the 6D pose (3D rotation and 3D translation) of objects from RGB images. The method uses:
- **Vector fields** to point from object pixels toward keypoints
- **Potential fields** to weight the voting process based on distance
- **Pixel-wise voting** to localize 2D keypoints
- **PnP algorithm** to recover 6D pose from 2D-3D keypoint correspondences

This implementation has been tested on the LINEMOD dataset, a standard benchmark for 6D pose estimation.

## Citation

This implementation is based on the following paper:

```bibtex
@article{lu2025dfwpvnet,
  title={DFW-PVNet: data field weighting based pixel-wise voting network for effective 6D pose estimation},
  author={Lu, Yilin and Pei, Shengchun},
  journal={Applied Intelligence},
  volume={55},
  pages={240},
  year={2025},
  doi={10.1007/s10489-024-05942-9},
  url={https://doi.org/10.1007/s10489-024-05942-9}
}
```

**Paper Reference:**
Lu, Y., Pei, S. DFW-PVNet: data field weighting based pixel-wise voting network for effective 6D pose estimation. *Appl Intell* **55**, 240 (2025). https://doi.org/10.1007/s10489-024-05942-9

## Project Structure

```
DFWPVNet_replication/
├── src/
│   ├── backbone.py                  # ResNet18 feature extraction backbone
│   ├── unet_decoder.py              # UNet-style decoder for feature fusion
│   ├── mask_head.py                 # Object segmentation head
│   ├── vector_field_head.py         # Vector field prediction head
│   ├── potential_field_head.py      # Potential field prediction head
│   ├── dfwpvnet.py                  # Main model architecture
│   ├── loss_functions.py            # Combined loss functions
│   ├── linemod_dataset.py           # LINEMOD dataset loader
│   ├── ground_truth_generator.py    # Generate vector & potential fields
│   ├── keypoint_selector.py         # FPS-based 3D keypoint selection
│   ├── inference/
│   │   ├── pose_estimator.py        # 6D pose estimation pipeline
│   │   └── keypoint_localizer.py    # 2D keypoint localization
│   └── evaluation/
│       └── metrics.py               # Evaluation metrics (ADD, ADD-S)
├── train.py                         # Training script with visualization
└── README.md

```

## Implemented Features

### Model Architecture
- **Backbone**: ResNet18 for multi-scale feature extraction
- **Decoder**: UNet-style architecture with skip connections
- **Multi-task Heads**:
  - Segmentation mask prediction
  - Vector field prediction (2K channels for K keypoints)
  - Potential field prediction (K channels for K keypoints)

### Training
- **Dataset**: LINEMOD dataset with 13 object classes
- **Training features**:
  - 200 epochs with learning rate scheduling
  - Adam optimizer with StepLR scheduler
  - Combined loss function (segmentation + vector field + potential field)
  - Checkpoint saving every 20 epochs
  - **Real-time loss visualization** with dynamic plotting

### Inference
- 2D keypoint localization using weighted pixel-wise voting
- PnP-based 6D pose recovery
- Support for RANSAC-based outlier rejection

### Evaluation
- ADD (Average Distance of Model Points) metric
- ADD-S (Symmetric object variant) metric

## Requirements

```
torch>=2.0.0
torchvision>=0.15.0
numpy
opencv-python
matplotlib
plyfile
scipy
```

## Dataset Setup

This implementation uses the LINEMOD dataset in BOP format:

1. Download the LINEMOD dataset from [BOP Challenge](https://bop.felk.cvut.cz/datasets/)
2. Extract to `dataset/linemod/linemod/`
3. Expected structure:
```
dataset/linemod/linemod/
├── models/                  # 3D object models (.ply files)
├── real/                    # Real images
│   ├── 000001/
│   │   ├── rgb/            # RGB images
│   │   ├── mask_visib/     # Segmentation masks
│   │   ├── scene_gt.json   # Ground truth poses
│   │   └── scene_camera.json
│   └── ...
└── {class_id}_train.txt     # Train/test splits
```

## Training

Run the training script:

```bash
python train.py
```

**Features during training:**
- Real-time loss plot window (updates dynamically)
- Loss plots saved to `training_loss.png` (updates every 5 epochs)
- Model checkpoints saved every 20 epochs
- Final plot saved as `final_training_loss.png`

**Training Configuration:**
- Batch size: 8
- Epochs: 200
- Learning rate: 0.001 (halved every 20 epochs)
- Number of keypoints: 9 (8 FPS + 1 center)
- Loss weights: λ₁=1.0 (mask), λ₂=0.1 (vector), λ₃=10.0 (potential)

## Model Details

The network predicts three outputs for each input image:

1. **Segmentation Mask** (H×W×2): Binary classification for object vs. background
2. **Vector Field** (H×W×2K): Unit vectors pointing toward K keypoints
3. **Potential Field** (H×W×K): Distance-based weights for voting

The 2D keypoints are then used with 3D keypoint correspondences to solve for 6D pose via PnP.

## Notes

- The implementation currently supports single-GPU training
- For multi-GPU training, consider using `torch.nn.DataParallel` or `DistributedDataParallel`
- Interactive loss visualization requires a display (TkAgg or Qt5Agg backend)
- On headless servers, plots are saved to files only

## Team

Ontario Tech University - AI & ML Course Project

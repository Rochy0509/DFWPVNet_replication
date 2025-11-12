import sys
import os
# Add parent directory to path so we can import from src
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.linemod_dataset import LINEMODDataset
import matplotlib.pyplot as plt

# Test dataset loading
data_dir = 'dataset/linemod/linemod'
print(f"Testing dataset at: {data_dir}")

try:
    dataset = LINEMODDataset(data_dir, split='train', num_keypoints=9)
    print(f"✅ Dataset loaded: {len(dataset)} samples")
    
    # Get first sample
    sample = dataset[0]
    print(f"✅ Sample keys: {sample.keys()}")
    print(f"   Image shape: {sample['image'].shape}")
    print(f"   Mask shape: {sample['mask'].shape}")
    print(f"   Vertex shape: {sample['vertex'].shape}")
    print(f"   Potential shape: {sample['potential'].shape}")
    
except Exception as e:
    print(f"❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
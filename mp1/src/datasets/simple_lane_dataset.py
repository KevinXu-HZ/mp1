import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

class SimpleLaneDataset(Dataset):
    def __init__(self, dataset_path, mode="train", size=(640, 384)):
        """
        Initialize the dataset.
        
        Args:
            dataset_path (str): Path to the dataset directory
            mode (str): 'train', 'test', or 'val'
            size (tuple): (width, height) for resizing images
        """
        assert mode in ["train", "test", "val"], "Mode must be 'train', 'test', or 'val'"
        
        self.dataset_path = os.path.join(dataset_path, mode)
        self.mode = mode
        self.size = size
        
        # Get all image files (excluding mask files)
        self.image_files = []
        for file in os.listdir(self.dataset_path):
            if file.endswith('.png') and not file.endswith('_mask.png'):
                self.image_files.append(file)
        
        print(f"Found {len(self.image_files)} images in {mode} set")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        """
        Get a single data sample.
        
        Returns:
            image (torch.Tensor): Input image (1 x H x W)
            mask (torch.Tensor): Segmentation mask (H x W)
        """
        # Get image filename
        img_name = self.image_files[idx]
        mask_name = img_name.replace('.png', '_mask.png')
        
        # Load image
        img_path = os.path.join(self.dataset_path, img_name)
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Failed to load image: {img_path}")
        
        # Load mask
        mask_path = os.path.join(self.dataset_path, mask_name)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Failed to load mask: {mask_path}")
        
        # Resize image and mask
        img = cv2.resize(img, self.size, interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, self.size, interpolation=cv2.INTER_NEAREST)
        
        # Convert image to grayscale and normalize
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img.astype(np.float32) / 255.0
        
        # Normalize mask values from 0-255 to class indices 0-1
        mask = (mask / 255).astype(np.int64)  # Convert to class indices
        
        # Convert to tensors
        img = torch.from_numpy(img).float().unsqueeze(0)  # Add channel dimension
        mask = torch.from_numpy(mask).long()  # Keep as integer for classification
        
        return img, mask

    def get_class_weights(self):
        """
        Calculate class weights for handling class imbalance.
        Returns weights for background and lane lines (2 classes).
        """
        class_counts = np.zeros(2)  # Background (0), lane lines (1)
        
        for idx in range(len(self)):
            _, mask = self[idx]
            for i in range(2):
                class_counts[i] += (mask == i).sum().item()
        
        # Calculate weights (inverse of frequency)
        total = class_counts.sum()
        weights = total / (2 * class_counts)
        weights = weights / weights.sum()  # Normalize
        
        return torch.from_numpy(weights).float()

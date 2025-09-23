#!/usr/bin/env python3
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import cv2
import numpy as np
from models.simple_enet import SimpleENet
from datasets.simple_lane_dataset import SimpleLaneDataset

def test_lane_detection():
    """
    Test the lane detection pipeline with the trained model.
    """
    print("Testing lane detection pipeline...")
    
    # Load the trained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleENet(num_classes=2).to(device)
    
    # Load the latest checkpoint
    checkpoint_path = "checkpoints/simple_enet_demo_epoch_10.pth"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print(f"Model loaded from {checkpoint_path}")
    else:
        print(f"Checkpoint not found: {checkpoint_path}")
        return
    
    # Load a test image from the dataset
    dataset = SimpleLaneDataset("data/dataset", mode="val")
    if len(dataset) == 0:
        print("No test images available")
        return
    
    # Test on a few images
    for i in range(min(3, len(dataset))):
        print(f"\nTesting image {i+1}/{min(3, len(dataset))}")
        
        # Get image and ground truth mask
        img_tensor, gt_mask = dataset[i]
        
        # Add batch dimension
        img_batch = img_tensor.unsqueeze(0).to(device)
        
        # Run inference
        with torch.no_grad():
            output = model(img_batch)
            pred_mask = torch.argmax(output, dim=1)
        
        # Convert to numpy for visualization
        img_np = img_tensor.squeeze().cpu().numpy()
        pred_np = pred_mask.squeeze().cpu().numpy()
        gt_np = gt_mask.cpu().numpy()
        
        # Calculate accuracy
        accuracy = (pred_np == gt_np).mean()
        print(f"Accuracy: {accuracy:.3f}")
        
        # Create visualization
        vis_img = np.zeros((img_np.shape[0], img_np.shape[1], 3), dtype=np.uint8)
        vis_img[..., 0] = img_np * 255  # Original image in red channel
        vis_img[..., 1] = pred_np * 127  # Predictions in green channel
        vis_img[..., 2] = gt_np * 127    # Ground truth in blue channel
        
        # Save visualization
        output_path = f"test_output_{i+1}.png"
        cv2.imwrite(output_path, vis_img)
        print(f"Visualization saved to {output_path}")
    
    print("\nLane detection test completed!")


if __name__ == '__main__':
    test_lane_detection()

#!/usr/bin/env python3
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import numpy as np
import wandb
from torch.utils.data import DataLoader
from datasets.simple_lane_dataset import SimpleLaneDataset
from models.simple_enet import SimpleENet
import torch.nn.functional as F
import tqdm
import cv2

# Configuration
BATCH_SIZE = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATASET_PATH = os.path.join("data", "dataset")
CHECKPOINT_PATH = "checkpoints/simple_enet_checkpoint_epoch_40.pth"  # Path to the trained model checkpoint

def evaluate():
    """
    Evaluate the trained SimpleENet model on a validation dataset and log results to Weights and Biases.
    """
    wandb.init(
        project="lane-detection",
        name="SimpleENet-Evaluation",
        config={
            "batch_size": BATCH_SIZE,
            "checkpoint_path": CHECKPOINT_PATH,
            "dataset_path": DATASET_PATH
        }
    )

    val_dataset = SimpleLaneDataset(DATASET_PATH, mode="val")
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model = SimpleENet(num_classes=2).to(DEVICE)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Checkpoint successfully loaded from {CHECKPOINT_PATH} (Epoch {checkpoint['epoch']})")

    model.eval()
    val_loss = 0
    i=0
    with torch.no_grad():
        for images, segmentation_labels in tqdm.tqdm(val_loader, desc="Evaluating"):
            images = images.to(DEVICE)
            segmentation_labels = segmentation_labels.to(DEVICE)

            # Forward pass
            outputs = model(images)
            
            # Compute loss
            loss = F.cross_entropy(outputs, segmentation_labels)
            val_loss += loss.item()

            # Convert predictions to segmentation mask
            pred_masks = torch.argmax(outputs, dim=1)
            
            # Create visualization for first batch
            if(i==0):
                i=1
                vis_images = []
                for i in range(min(4, images.size(0))):  # Show first 4 images
                    img = images[i].cpu().numpy().squeeze()
                    pred = pred_masks[i].cpu().numpy()
                    label = segmentation_labels[i].cpu().numpy()
                    
                    # Create BGR visualization
                    vis = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
                    vis[..., 0] = img * 127  # Original image in blue channel
                    vis[..., 1] = pred * 255  # Predictions in green channel
                    vis[..., 2] = label * 127  # Ground truth in red channel

                    vis_images.append(vis)
                
                # Stack images horizontally
                combined_row = np.hstack(vis_images)

                cv2.imwrite("validation_row.png", combined_row)
                wandb.log({"visualization": wandb.Image(combined_row, caption="Evaluation Visualization")})

    mean_loss = val_loss / len(val_loader)
    wandb.log({
        "val_loss": mean_loss
    })

    print(f"Evaluation Results:\n"
          f"  - Validation Loss: {mean_loss:.4f}")

    wandb.finish()

if __name__ == '__main__':
    evaluate()
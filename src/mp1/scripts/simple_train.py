#!/usr/bin/env python3
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import tqdm
import wandb
import torch
import numpy as np
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from models.simple_enet import SimpleENet
from datasets.simple_lane_dataset import SimpleLaneDataset


# Configurations
##### YOUR CODE STARTS HERE #####
BATCH_SIZE = 
LR = 
EPOCHS = 
##### YOUR CODE ENDS HERE #####
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATASET_PATH = "data/dataset"
CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def validate(model, val_loader):
    """
    Validate the model on the validation dataset.
    """
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for images, segmentation_labels in tqdm.tqdm(val_loader, desc="Validating"):
            images = images.to(DEVICE)
            segmentation_labels = segmentation_labels.to(DEVICE)

            # Forward pass
            outputs = model(images)
            
            # Compute loss (Cross Entropy for segmentation)
            loss = F.cross_entropy(outputs, segmentation_labels)
            val_loss += loss.item()

    return val_loss / len(val_loader)

def train():
    """
    Train the SimpleENet model on the training dataset and log results to Weights & Biases.
    """
    wandb.init(
        project="lane-detection",
        name="SimpleENet-Training",
        config={
            "batch_size": BATCH_SIZE,
            "learning_rate": LR,
            "epochs": EPOCHS,
            "optimizer": "Adam",
            "model": "SimpleENet"
        }
    )

    # Data preparation
    train_dataset = SimpleLaneDataset(DATASET_PATH, mode="train")
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    val_dataset = SimpleLaneDataset(DATASET_PATH, mode="test")
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Model and optimizer initialization
    model = SimpleENet(num_classes=2).to(DEVICE)
    ##### YOUR CODE STARTS HERE #####
   
    ##### YOUR CODE ENDS HERE #####
    

    def save_checkpoint(model, optimizer, epoch, checkpoint_dir):
        """
        Save model checkpoints during training.
        """
        checkpoint_path = os.path.join(checkpoint_dir, f"simple_enet_checkpoint_epoch_{epoch}.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, checkpoint_path)
        wandb.save(checkpoint_path)
        print(f"Checkpoint saved at {checkpoint_path}")

    # Training loop
    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0

        for batch_idx, (images, segmentation_labels) in enumerate(tqdm.tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")):
            
            """
            Implement the training loop in one epoch

            Steps:
                1. Move images, segmentation_labels to the desired device
                2. Feed the images to the model and get the predictions
                3. Compare the predictions with the ground truth using cross entropy loss
                4. Clear the existing gradient, do the backpropagation, and update the weights
            """

            ##### YOUR CODE STARTS HERE #####

            # Move data to device
          
            # Forward pass
             
            # Compute loss
          
            # Backward pass and optimize
   
            ##### YOUR CODE ENDS HERE #####
            

            epoch_loss += loss.item()

            # Log visualizations for the first batch of the epoch
            if batch_idx == 0:
                # Convert predictions to segmentation mask
                pred_masks = torch.argmax(outputs, dim=1)
                
                # Create visualization
                vis_images = []
                for i in range(min(4, images.size(0))):  # Show first 4 images
                    img = images[i].cpu().numpy().squeeze()
                    pred = pred_masks[i].cpu().numpy()
                    label = segmentation_labels[i].cpu().numpy()
                    
                    # Create RGB visualization
                    vis = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
                    vis[..., 0] = img * 255  # Original image in red channel
                    vis[..., 1] = pred * 127  # Predictions in green channel
                    vis[..., 2] = label * 127  # Ground truth in blue channel
                    
                    vis_images.append(vis)
                
                # Stack images horizontally
                combined_row = np.hstack(vis_images)
                wandb.log({"visualization": wandb.Image(combined_row, caption=f"Epoch {epoch} - Batch {batch_idx}")})

        # Epoch-wise logging
        mean_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch}/{EPOCHS}: Loss = {mean_loss:.4f}")
        wandb.log({
            "epoch": epoch,
            "train_loss": mean_loss
        })

        # Validation
        val_loss = validate(model, val_loader)
        print(f"Validation Loss = {val_loss:.4f}")
        wandb.log({
            "val_loss": val_loss
        })

        save_checkpoint(model, optimizer, epoch, CHECKPOINT_DIR)

    wandb.finish()

if __name__ == '__main__':
    train()

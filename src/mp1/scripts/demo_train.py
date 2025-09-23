#!/usr/bin/env python3
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import tqdm
import torch
import numpy as np
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from models.simple_enet import SimpleENet
from datasets.simple_lane_dataset import SimpleLaneDataset


# Configurations for demo training
BATCH_SIZE = 4
LR = 0.001
EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATASET_PATH = "data/dataset"
CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

print(f"Using device: {DEVICE}")


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
    Train the SimpleENet model on the available dataset.
    """
    print("Starting demo training...")
    
    # Data preparation - use val data for both train and val since we only have val data
    train_dataset = SimpleLaneDataset(DATASET_PATH, mode="val")
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    
    val_dataset = SimpleLaneDataset(DATASET_PATH, mode="val")
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Model and optimizer initialization
    model = SimpleENet(num_classes=2).to(DEVICE)
    optimizer = Adam(model.parameters(), lr=LR)
    
    print(f"Training on {len(train_dataset)} samples")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    def save_checkpoint(model, optimizer, epoch, checkpoint_dir):
        """
        Save model checkpoints during training.
        """
        checkpoint_path = os.path.join(checkpoint_dir, f"simple_enet_demo_epoch_{epoch}.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, checkpoint_path)
        print(f"Checkpoint saved at {checkpoint_path}")

    # Training loop
    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0

        for batch_idx, (images, segmentation_labels) in enumerate(tqdm.tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")):
            
            # Move data to device
            images = images.to(DEVICE)
            segmentation_labels = segmentation_labels.to(DEVICE)
            
            # Forward pass
            outputs = model(images)
            
            # Compute loss
            loss = F.cross_entropy(outputs, segmentation_labels)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()

        # Epoch-wise logging
        mean_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch}/{EPOCHS}: Loss = {mean_loss:.4f}")

        # Validation
        val_loss = validate(model, val_loader)
        print(f"Validation Loss = {val_loss:.4f}")

        # Save checkpoint every few epochs
        if epoch % 5 == 0 or epoch == EPOCHS:
            save_checkpoint(model, optimizer, epoch, CHECKPOINT_DIR)

    print("Training completed!")


if __name__ == '__main__':
    train()

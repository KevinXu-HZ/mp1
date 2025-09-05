#!/usr/bin/env python3
import os
import sys
import torch
import numpy as np
import wandb
import rclpy
from rclpy.node import Node
from torch.utils.data import DataLoader
from datasets.simple_lane_dataset import SimpleLaneDataset
from models.simple_enet import SimpleENet
import torch.nn.functional as F
import tqdm
import cv2
from std_srvs.srv import Trigger

# Configuration
BATCH_SIZE = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ModelEvaluationService(Node):
    def __init__(self):
        super().__init__('model_evaluation_service')
        
        # Declare parameters
        self.declare_parameter('dataset_path', 
                              os.path.join(os.path.dirname(os.path.realpath(__file__)), "dataset"))
        self.declare_parameter('checkpoint_path', "checkpoints/simple_enet_checkpoint_epoch_20.pth")
        self.declare_parameter('batch_size', 10)
        self.declare_parameter('use_wandb', True)
        
        # Get parameters
        self.dataset_path = self.get_parameter('dataset_path').get_parameter_value().string_value
        self.checkpoint_path = self.get_parameter('checkpoint_path').get_parameter_value().string_value
        self.batch_size = self.get_parameter('batch_size').get_parameter_value().integer_value
        self.use_wandb = self.get_parameter('use_wandb').get_parameter_value().bool_value
        
        # Create service
        self.srv = self.create_service(Trigger, 'evaluate_model', self.evaluate_callback)
        
        self.get_logger().info('Model evaluation service ready')
        self.get_logger().info(f'Dataset path: {self.dataset_path}')
        self.get_logger().info(f'Checkpoint path: {self.checkpoint_path}')
        self.get_logger().info(f'Batch size: {self.batch_size}')
        self.get_logger().info(f'Device: {DEVICE}')

    def evaluate_callback(self, request, response):
        """
        Service callback to evaluate the trained SimpleENet model.
        """
        try:
            self.get_logger().info('Starting model evaluation...')
            
            # Initialize wandb if enabled
            if self.use_wandb:
                wandb.init(
                    project="lane-detection",
                    name="SimpleENet-Evaluation-ROS2",
                    config={
                        "batch_size": self.batch_size,
                        "checkpoint_path": self.checkpoint_path,
                        "dataset_path": self.dataset_path
                    }
                )

            # Load dataset
            if not os.path.exists(self.dataset_path):
                error_msg = f"Dataset path does not exist: {self.dataset_path}"
                self.get_logger().error(error_msg)
                response.success = False
                response.message = error_msg
                return response

            val_dataset = SimpleLaneDataset(self.dataset_path, mode="test")
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)
            self.get_logger().info(f"Loaded dataset with {len(val_dataset)} test samples")

            # Load model
            if not os.path.exists(self.checkpoint_path):
                error_msg = f"Checkpoint file does not exist: {self.checkpoint_path}"
                self.get_logger().error(error_msg)
                response.success = False
                response.message = error_msg
                return response

            model = SimpleENet(num_classes=2).to(DEVICE)
            checkpoint = torch.load(self.checkpoint_path, map_location=DEVICE)
            model.load_state_dict(checkpoint['model_state_dict'])
            self.get_logger().info(f"Checkpoint successfully loaded from {self.checkpoint_path} (Epoch {checkpoint['epoch']})")

            # Evaluate model
            model.eval()
            val_loss = 0
            total_samples = 0
            visualizations_saved = 0
            
            with torch.no_grad():
                for batch_idx, (images, segmentation_labels) in enumerate(tqdm.tqdm(val_loader, desc="Evaluating")):
                    images = images.to(DEVICE)
                    segmentation_labels = segmentation_labels.to(DEVICE)

                    # Forward pass
                    outputs = model(images)
                    
                    # Compute loss
                    loss = F.cross_entropy(outputs, segmentation_labels)
                    val_loss += loss.item()
                    total_samples += images.size(0)

                    # Convert predictions to segmentation mask
                    pred_masks = torch.argmax(outputs, dim=1)
                    
                    # Create visualization for first batch only
                    if batch_idx == 0 and visualizations_saved == 0:
                        visualizations_saved = 1
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
                        if vis_images:
                            combined_row = np.hstack(vis_images)
                            
                            # Save visualization locally
                            viz_path = os.path.join(os.path.dirname(__file__), 'evaluation_visualization.png')
                            cv2.imwrite(viz_path, combined_row)
                            self.get_logger().info(f"Saved evaluation visualization to: {viz_path}")
                            
                            # Log to wandb if enabled
                            if self.use_wandb:
                                wandb.log({"visualization": wandb.Image(combined_row, caption="Evaluation Visualization")})

            mean_loss = val_loss / len(val_loader)
            
            # Log results
            if self.use_wandb:
                wandb.log({"val_loss": mean_loss})
                wandb.finish()

            result_message = (f"Evaluation Results:\n"
                            f"  - Validation Loss: {mean_loss:.4f}\n"
                            f"  - Total samples: {total_samples}\n"
                            f"  - Device used: {DEVICE}")
            
            self.get_logger().info(result_message)
            
            response.success = True
            response.message = result_message

        except Exception as e:
            error_msg = f"Error during evaluation: {str(e)}"
            self.get_logger().error(error_msg)
            response.success = False
            response.message = error_msg
            
            if self.use_wandb:
                wandb.finish()

        return response


def main(args=None):
    rclpy.init(args=args)
    
    try:
        evaluation_service = ModelEvaluationService()
        rclpy.spin(evaluation_service)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'evaluation_service' in locals():
            evaluation_service.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

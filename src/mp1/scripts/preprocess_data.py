#!/usr/bin/env python3
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_dir, '..', 'src'))

import cv2
import torch
import random
import numpy as np
from PIL import Image as PILImage
from utils.ground_truth_generator import GroundTruthGenerator


def process_dataset(image_dir: str = "data/raw_images/", output_dir: str = "data/dataset/", train_ratio: float = 0.8):
    world_fn = os.path.join(script_dir,'..','..','gem_simulator/gem_gazebo/worlds/smaller_track.world')
    generator = GroundTruthGenerator(world_fn=world_fn, resolution=20)
    train_dir = os.path.join(output_dir, 'train')
    test_dir = os.path.join(output_dir, 'test')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(train_dir):
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)
    # prepare all image files
    image_files = []
    for root, _, files in os.walk(image_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_files.append(os.path.join(root, file))
    random.shuffle(image_files)
    # split into train/test
    split_idx = int(len(image_files) * train_ratio)
    train_files = image_files[:split_idx]
    test_files = image_files[split_idx:]
    # process training set
    print("Processing training set...")
    for img_path in train_files:
        process_image(img_path, train_dir, generator=generator)
    # process test set
    print("Processing test set...")
    for img_path in test_files:
        process_image(img_path, test_dir, generator=generator)
    print(f"Dataset processing complete. {len(train_files)} training images, {len(test_files)} test images.")

def process_image(img_path, output_dir, generator):
    """
    Process a single image and generate its segmentation mask using the lane detector.
    """
    # obtain vehicle position and yaw at times of image capture
    try:
        metadata = PILImage.open(img_path).info
        img = cv2.imread(img_path)
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        mask = generator.generate_ground_truth([float(metadata["vehicle_x"]), float(metadata["vehicle_y"]), float(metadata["vehicle_yaw"])])
        cv2.imwrite(os.path.join(output_dir, f"{img_name}.png"), img)
        cv2.imwrite(os.path.join(output_dir, f"{img_name}_mask.png"), mask)
    except Exception as e:
        print("Failed to load metadata from PILImage with error:", e)


if __name__ == "__main__":
    process_dataset()

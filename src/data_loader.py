"""
Embryo Dataset and DataLoader implementation for the Embryo Analyzer project.

This module provides a custom PyTorch Dataset class for loading embryo images
and a function to create training and validation DataLoaders with appropriate
transformations.
"""

import os
from typing import Tuple, List, Optional, Dict, Any
from pathlib import Path

import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np


class EmbryoDataset(Dataset):
    """
    Custom PyTorch Dataset for loading embryo images organized by developmental stage.

    This dataset automatically discovers images from a directory structure where:
    - Root directory contains stage folders (e.g., '2cell', '4cell', '8cell', etc.)
    - Each stage folder contains 'train' and 'test' subdirectories
    - Images are stored as PNG files within these subdirectories

    The class labels are automatically inferred from the stage folder names.

    Attributes:
        root_dir (str): Path to the root directory containing stage folders
        split (str): Data split to use ('train' or 'test')
        transforms (torchvision.transforms.Compose): Image transformations to apply
        image_paths (List[str]): List of all image file paths
        labels (List[int]): List of corresponding numeric labels
        class_to_idx (Dict[str, int]): Mapping from class names to numeric indices
    """

    def __init__(self, root_dir: str, split: str, transforms: Optional[transforms.Compose] = None) -> None:
        """
        Initialize the EmbryoDataset.

        Args:
            root_dir (str): Path to the root directory containing stage folders
                           (e.g., 'data/processed/')
            split (str): Data split to use ('train' or 'test')
            transforms (torchvision.transforms.Compose, optional): Image transformations to apply.
                                                                  If None, only basic tensor conversion is applied.

        Raises:
            ValueError: If split is not 'train' or 'test'
            FileNotFoundError: If root_dir does not exist
        """
        if split not in ['train', 'test']:
            raise ValueError("split must be either 'train' or 'test'")

        self.root_dir = Path(root_dir)
        self.split = split
        self.transforms = transforms

        if not self.root_dir.exists():
            raise FileNotFoundError(
                f"Root directory does not exist: {self.root_dir}")

        # Discover all stage directories (excluding any hidden files/folders)
        stage_dirs = [d for d in self.root_dir.iterdir()
                      if d.is_dir() and not d.name.startswith('.')]

        if not stage_dirs:
            raise ValueError(f"No stage directories found in {self.root_dir}")

        # Create class mapping from stage names to indices
        self.class_to_idx = {stage_dir.name: idx for idx,
                             stage_dir in enumerate(sorted(stage_dirs))}

        # Collect all image paths and their labels
        self.image_paths: List[str] = []
        self.labels: List[int] = []

        for stage_dir in stage_dirs:
            split_dir = stage_dir / split
            if not split_dir.exists():
                print(
                    f"Warning: Split directory {split_dir} does not exist, skipping...")
                continue

            # Find all PNG image files in this split directory
            image_files = list(split_dir.glob("*.png"))
            if not image_files:
                print(f"Warning: No PNG files found in {split_dir}")
                continue

            # Add image paths and corresponding labels
            for image_file in image_files:
                self.image_paths.append(str(image_file))
                self.labels.append(self.class_to_idx[stage_dir.name])

        if not self.image_paths:
            raise ValueError(
                f"No images found for split '{split}' in {self.root_dir}")

        print(f"Dataset initialized for {split} split:")
        print(
            f"  - Found {len(self.class_to_idx)} classes: {list(self.class_to_idx.keys())}")
        print(f"  - Total images: {len(self.image_paths)}")

    def __len__(self) -> int:
        """
        Return the total number of images in the dataset.

        Returns:
            int: Number of images in the dataset
        """
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Load and return an image and its label at the given index.

        Args:
            idx (int): Index of the image to retrieve

        Returns:
            Tuple[torch.Tensor, int]: A tuple containing:
                - The transformed image as a PyTorch tensor
                - The numeric label of the image

        Raises:
            IndexError: If idx is out of bounds
        """
        if idx < 0 or idx >= len(self.image_paths):
            raise IndexError(
                f"Index {idx} is out of bounds for dataset of size {len(self.image_paths)}")

        # Load image using OpenCV
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise ValueError(f"Failed to load image at {image_path}")

        # Convert to RGB format (3 channels) for torchvision transforms
        # Most pre-trained models expect 3-channel RGB images
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # Convert to PIL Image for torchvision transforms
        from PIL import Image
        image = Image.fromarray(image)

        # Apply transformations if provided
        if self.transforms is not None:
            image = self.transforms(image)

        # Get the corresponding label
        label = self.labels[idx]

        return image, label

    @property
    def classes(self) -> List[str]:
        """
        Get the list of class names in sorted order.

        Returns:
            List[str]: List of class names
        """
        return sorted(self.class_to_idx.keys())

    def get_class_distribution(self) -> Dict[str, int]:
        """
        Get the distribution of images across classes.

        Returns:
            Dict[str, int]: Dictionary mapping class names to their image counts
        """
        distribution = {}
        for class_name in self.classes:
            class_idx = self.class_to_idx[class_name]
            count = self.labels.count(class_idx)
            distribution[class_name] = count
        return distribution


def create_dataloaders(
    root_dir: str,
    batch_size: int = 32,
    image_size: Tuple[int, int] = (224, 224),
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation DataLoaders for the embryo dataset.

    This function sets up appropriate data transformations for training and validation,
    creates EmbryoDataset instances for both splits, and wraps them in DataLoaders.

    Args:
        root_dir (str): Path to the root directory containing stage folders
        batch_size (int): Number of images per batch. Defaults to 32.
        image_size (Tuple[int, int]): Target image size (height, width). Defaults to (224, 224).
        num_workers (int): Number of worker processes for data loading. Defaults to 0
                          (use 0 for Windows compatibility).

    Returns:
        Tuple[DataLoader, DataLoader]: A tuple containing:
            - Training DataLoader with data augmentation and shuffling
            - Validation DataLoader without augmentation and no shuffling

    Raises:
        ValueError: If batch_size <= 0 or image_size dimensions are invalid
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")

    if len(image_size) != 2 or image_size[0] <= 0 or image_size[1] <= 0:
        raise ValueError(
            "image_size must be a tuple of two positive integers (height, width)")

    # Define training transformations with data augmentation
    train_transforms = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(p=0.5),  # Random horizontal flip
        # Random rotation up to 10 degrees
        transforms.RandomRotation(degrees=10),
        # Random brightness/contrast
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])  # ImageNet normalization
    ])

    # Define validation transformations without augmentation
    val_transforms = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])  # ImageNet normalization
    ])

    # Create datasets
    try:
        train_dataset = EmbryoDataset(
            root_dir=root_dir, split='train', transforms=train_transforms)
        val_dataset = EmbryoDataset(
            root_dir=root_dir, split='test', transforms=val_transforms)
    except Exception as e:
        raise RuntimeError(f"Failed to create datasets: {str(e)}")

    # Print dataset information
    print(f"\nTraining set: {len(train_dataset)} images")
    print(f"Validation set: {len(val_dataset)} images")
    print(
        f"Class distribution (training): {train_dataset.get_class_distribution()}")

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Shuffle training data
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()  # Use pinned memory if GPU is available
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # Don't shuffle validation data
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()  # Use pinned memory if GPU is available
    )

    return train_loader, val_loader

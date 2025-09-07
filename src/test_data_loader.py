#!/usr/bin/env python3
"""
Quick test script for the data loader module.

This script can be run from the command line to verify that the data pipeline
is working correctly without needing to open a Jupyter notebook.

Usage:
    python src/test_data_loader.py
"""

from data_loader import create_dataloaders, EmbryoDataset
import sys
import os
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(__file__))


def main():
    """Test the data loader functionality."""
    print("Testing Embryo Data Loader")
    print("=" * 40)

    # Configuration
    data_dir = "../data/processed/"
    batch_size = 8  # Small batch for testing
    image_size = (224, 224)

    print(f"Data directory: {data_dir}")
    print(f"Batch size: {batch_size}")
    print(f"Image size: {image_size}")
    print()

    try:
        # Test dataset creation
        print("1. Testing dataset creation...")
        train_dataset = EmbryoDataset(root_dir=data_dir, split='train')
        val_dataset = EmbryoDataset(root_dir=data_dir, split='test')

        print(f"   ✓ Training dataset: {len(train_dataset)} images")
        print(f"   ✓ Validation dataset: {len(val_dataset)} images")
        print(f"   ✓ Classes: {train_dataset.classes}")
        print()

        # Test dataloader creation
        print("2. Testing dataloader creation...")
        train_loader, val_loader = create_dataloaders(
            root_dir=data_dir,
            batch_size=batch_size,
            image_size=image_size
        )
        print("   ✓ Dataloaders created successfully")
        print()

        # Test batch loading
        print("3. Testing batch loading...")
        train_images, train_labels = next(iter(train_loader))
        val_images, val_labels = next(iter(val_loader))

        print(f"   ✓ Training batch shape: {train_images.shape}")
        print(f"   ✓ Training labels shape: {train_labels.shape}")
        print(f"   ✓ Validation batch shape: {val_images.shape}")
        print(f"   ✓ Validation labels shape: {val_labels.shape}")
        print(
            f"   ✓ Image value range: [{train_images.min():.3f}, {train_images.max():.3f}]")
        print()

        # Test individual image loading
        print("4. Testing individual image loading...")
        image, label = train_dataset[0]
        # PIL Image .size is a property
        print(f"   ✓ Single image size: {image.size}")
        print(f"   ✓ Single label: {label}")
        print(
            f"   ✓ Label corresponds to class: {train_dataset.classes[label]}")
        print()

        print("🎉 All tests passed! The data pipeline is working correctly.")
        return True

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        print()
        print("Troubleshooting tips:")
        print("- Make sure the data directory exists and contains the expected structure")
        print(
            "- Check that all stage directories have both 'train' and 'test' subdirectories")
        print("- Verify that PNG image files exist in the subdirectories")
        print("- Ensure all required dependencies are installed (torch, torchvision, opencv-python-headless)")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

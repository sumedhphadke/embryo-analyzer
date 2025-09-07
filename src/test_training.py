#!/usr/bin/env python3
"""
Test script for the Embryo Analyzer training pipeline.

This script runs a quick test of the training pipeline with minimal epochs
to verify that all components work correctly together.
"""

import os
import sys
import torch

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(__file__))

def test_training_pipeline():
    """Test the complete training pipeline."""
    print("🧪 Testing Embryo Analyzer Training Pipeline")
    print("=" * 50)

    # Check if CUDA is available
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
    print()

    # Test imports
    print("Testing imports...")
    try:
        from data_loader import create_dataloaders
        from model import create_resnet50_model
        import yaml
        print("✅ All imports successful")
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

    # Test configuration loading
    print("\\nTesting configuration loading...")
    try:
        with open('test_config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        print("✅ Configuration loaded successfully")
        print(f"   Model: {config['model']['name']}")
        print(f"   Epochs: {config['training']['num_epochs']}")
        print(f"   Batch size: {config['data']['batch_size']}")
    except Exception as e:
        print(f"❌ Configuration loading failed: {e}")
        return False

    # Test dataloader creation
    print("\\nTesting dataloader creation...")
    try:
        train_loader, val_loader = create_dataloaders(
            root_dir=config['data']['root_dir'],
            batch_size=config['data']['batch_size'],
            image_size=tuple(config['data']['image_size'])
        )
        print("✅ Dataloaders created successfully")
        print(f"   Training batches: {len(train_loader)}")
        print(f"   Validation batches: {len(val_loader)}")
    except Exception as e:
        print(f"❌ Dataloader creation failed: {e}")
        return False

    # Test model creation
    print("\\nTesting model creation...")
    try:
        model = create_resnet50_model(
            num_classes=config['model']['num_classes'],
            pretrained=config['model']['pretrained']
        )
        print("✅ Model created successfully")

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return False

    # Test a forward pass
    print("\\nTesting forward pass...")
    try:
        model.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)

        # Get one batch
        images, labels = next(iter(train_loader))
        images = images.to(device)

        with torch.no_grad():
            outputs = model(images)

        print("✅ Forward pass successful")
        print(f"   Input shape: {images.shape}")
        print(f"   Output shape: {outputs.shape}")
        print(f"   Expected classes: {config['model']['num_classes']}")
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        return False

    print("\\n" + "=" * 50)
    print("🎉 All tests passed! The training pipeline is ready.")
    print("You can now run the full training with:")
    print("  python src/train.py")
    print("=" * 50)

    return True


if __name__ == "__main__":
    success = test_training_pipeline()
    sys.exit(0 if success else 1)

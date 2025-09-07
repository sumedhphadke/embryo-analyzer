#!/usr/bin/env python3
"""
Embryo Analyzer Training Script

This script trains a deep learning model to classify embryo developmental stages
from microscope images. It uses a ResNet50 architecture fine-tuned for the task.

Usage:
    python src/train.py

The script reads configuration from src/config.yaml and saves the best model
to the models/ directory.
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import yaml

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim import lr_scheduler

# Import our custom modules
from data_loader import create_dataloaders
from model import create_resnet50_model, create_optimizer, create_scheduler, get_model_summary


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path (str): Path to the configuration file

    Returns:
        Dict[str, Any]: Configuration dictionary

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is malformed
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print(f"Configuration loaded from {config_path}")
    return config


def get_device(device_config: str) -> torch.device:
    """
    Get the appropriate device for training.

    Args:
        device_config (str): Device configuration ('auto', 'cpu', 'cuda', 'mps')

    Returns:
        torch.device: PyTorch device object
    """
    if device_config == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print("Using CUDA GPU for training")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
            print("Using Apple Silicon MPS for training")
        else:
            device = torch.device('cpu')
            print("Using CPU for training")
    else:
        device = torch.device(device_config)
        print(f"Using specified device: {device_config}")

    return device


def create_loss_function(loss_config: Dict[str, Any]) -> nn.Module:
    """
    Create the loss function based on configuration.

    Args:
        loss_config (Dict[str, Any]): Loss function configuration

    Returns:
        nn.Module: Configured loss function
    """
    loss_type = loss_config.get('type', 'cross_entropy')

    if loss_type == 'cross_entropy':
        label_smoothing = loss_config.get('label_smoothing', 0.0)
        loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        print(
            f"Using CrossEntropyLoss with label smoothing: {label_smoothing}")
    else:
        raise ValueError(f"Unsupported loss function: {loss_type}")

    return loss_fn


def create_optimizer_from_config(
    model: nn.Module,
    optimizer_config: Dict[str, Any],
    lr: float,
    weight_decay: float
) -> torch.optim.Optimizer:
    """
    Create optimizer based on configuration.

    Args:
        model (nn.Module): The model to optimize
        optimizer_config (Dict[str, Any]): Optimizer configuration
        lr (float): Learning rate
        weight_decay (float): Weight decay (L2 regularization)

    Returns:
        torch.optim.Optimizer: Configured optimizer
    """
    optimizer_type = optimizer_config.get('type', 'adam')

    # Only optimize parameters that require gradients (unfrozen layers)
    params_to_optimize = list(
        filter(lambda p: p.requires_grad, model.parameters()))

    if optimizer_type == 'adam':
        adam_config = optimizer_config.get('adam', {})
        optimizer = optim.Adam(
            params_to_optimize,
            lr=lr,
            weight_decay=weight_decay,
            betas=(adam_config.get('beta1', 0.9),
                   adam_config.get('beta2', 0.999)),
            eps=adam_config.get('eps', 1e-8)
        )
    elif optimizer_type == 'sgd':
        momentum = optimizer_config.get('momentum', 0.9)
        optimizer = optim.SGD(
            params_to_optimize,
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum
        )
    elif optimizer_type == 'adamw':
        optimizer = optim.AdamW(
            params_to_optimize,
            lr=lr,
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_type}")

    print(
        f"Created {optimizer_type.upper()} optimizer with learning rate: {lr}")
    return optimizer


def create_scheduler_from_config(
    optimizer: torch.optim.Optimizer,
    scheduler_config: Dict[str, Any]
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """
    Create learning rate scheduler based on configuration.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer
        scheduler_config (Dict[str, Any]): Scheduler configuration

    Returns:
        Optional[torch.optim.lr_scheduler._LRScheduler]: Configured scheduler or None
    """
    scheduler_type = scheduler_config.get('type', 'step')

    if scheduler_type == 'step':
        step_size = scheduler_config.get('step_size', 7)
        gamma = scheduler_config.get('gamma', 0.1)
        scheduler = lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=gamma)
        print(
            f"Created StepLR scheduler with step_size={step_size}, gamma={gamma}")
    elif scheduler_type == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        print("Created CosineAnnealingLR scheduler")
    elif scheduler_type == 'exponential':
        gamma = scheduler_config.get('gamma', 0.9)
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
        print(f"Created ExponentialLR scheduler with gamma={gamma}")
    else:
        print(f"No scheduler configured for type: {scheduler_type}")
        return None

    return scheduler


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    config: Dict[str, Any]
) -> Tuple[float, float]:
    """
    Train the model for one epoch.

    Args:
        model (nn.Module): The model to train
        dataloader (DataLoader): Training data loader
        criterion (nn.Module): Loss function
        optimizer (torch.optim.Optimizer): Optimizer
        device (torch.device): Device to train on
        epoch (int): Current epoch number
        config (Dict[str, Any]): Training configuration

    Returns:
        Tuple[float, float]: Average training loss and accuracy
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    log_interval = config['logging']['log_interval']

    for batch_idx, (inputs, labels) in enumerate(dataloader):
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Log progress
        if (batch_idx + 1) % log_interval == 0:
            batch_loss = running_loss / (batch_idx + 1)
            batch_acc = 100. * correct / total
            print(f"Epoch [{epoch+1}], Batch [{batch_idx+1}/{len(dataloader)}], "
                  ".4f"
                  ".2f")

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total

    return epoch_loss, epoch_acc


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """
    Validate the model on the validation set.

    Args:
        model (nn.Module): The model to validate
        dataloader (DataLoader): Validation data loader
        criterion (nn.Module): Loss function
        device (torch.device): Device to validate on

    Returns:
        Tuple[float, float]: Validation loss and accuracy
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    val_loss = running_loss / len(dataloader)
    val_acc = 100. * correct / total

    return val_loss, val_acc


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    epoch: int,
    val_acc: float,
    save_dir: str,
    filename: str
) -> None:
    """
    Save a model checkpoint.

    Args:
        model (nn.Module): The model to save
        optimizer (torch.optim.Optimizer): The optimizer state
        scheduler (Optional[torch.optim.lr_scheduler._LRScheduler]): The scheduler state
        epoch (int): Current epoch
        val_acc (float): Validation accuracy
        save_dir (str): Directory to save the checkpoint
        filename (str): Checkpoint filename
    """
    os.makedirs(save_dir, exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_accuracy': val_acc,
    }

    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()

    checkpoint_path = os.path.join(save_dir, filename)
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")


def save_best_model(
    model: nn.Module,
    save_dir: str,
    filename: str,
    config: Dict[str, Any]
) -> None:
    """
    Save the best model weights.

    Args:
        model (nn.Module): The model to save
        save_dir (str): Directory to save the model
        filename (str): Model filename
        config (Dict[str, Any]): Configuration
    """
    os.makedirs(save_dir, exist_ok=True)

    model_path = os.path.join(save_dir, filename)

    # Save model state dict
    torch.save(model.state_dict(), model_path)
    print(f"Best model saved: {model_path}")

    # Also save configuration for reproducibility
    config_path = os.path.splitext(model_path)[0] + '_config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"Configuration saved: {config_path}")


def main():
    """Main training function."""
    print("🚀 Starting Embryo Analyzer Training")
    print("=" * 50)

    # Load configuration
    import argparse
    parser = argparse.ArgumentParser(description='Train Embryo Analyzer')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to configuration file')
    args = parser.parse_args()

    config_path = args.config
    try:
        config = load_config(config_path)
    except Exception as e:
        print(f"❌ Failed to load configuration: {e}")
        return

    # Set random seed for reproducibility
    seed = config.get('seed', 42)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    print(f"Random seed set to: {seed}")

    # Get device
    device = get_device(config.get('device', 'auto'))
    print(f"Using device: {device}")

    # Create dataloaders
    print("\\n📊 Setting up data loaders...")
    try:
        train_loader, val_loader = create_dataloaders(
            root_dir=config['data']['root_dir'],
            batch_size=config['data']['batch_size'],
            image_size=tuple(config['data']['image_size']),
            num_workers=config['data']['num_workers']
        )
    except Exception as e:
        print(f"❌ Failed to create dataloaders: {e}")
        return

    # Create model
    print("\\n🏗️  Creating model...")
    try:
        model = create_resnet50_model(
            num_classes=config['model']['num_classes'],
            pretrained=config['model']['pretrained']
        )
        model = model.to(device)

        # Print model summary
        print(get_model_summary(model, input_size=(
            3, *config['data']['image_size'])))

    except Exception as e:
        print(f"❌ Failed to create model: {e}")
        return

    # Create loss function
    print("\\n📉 Setting up loss function...")
    try:
        criterion = create_loss_function(config['loss'])
        criterion = criterion.to(device)
    except Exception as e:
        print(f"❌ Failed to create loss function: {e}")
        return

    # Create optimizer
    print("\\n⚡ Setting up optimizer...")
    print(f"   Optimizer config: {config['optimizer']}")
    print(
        f"   Learning rate: {config['training']['learning_rate']} (type: {type(config['training']['learning_rate'])})")
    print(
        f"   Weight decay: {config['training']['weight_decay']} (type: {type(config['training']['weight_decay'])})")
    try:
        optimizer = create_optimizer_from_config(
            model,
            config['optimizer'],
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
    except Exception as e:
        print(f"❌ Failed to create optimizer: {e}")
        import traceback
        traceback.print_exc()
        return

    # Create scheduler
    print("\\n📅 Setting up learning rate scheduler...")
    try:
        scheduler = create_scheduler_from_config(
            optimizer, config['scheduler'])
    except Exception as e:
        print(f"❌ Failed to create scheduler: {e}")
        return

    # Training loop
    print("\\n🎯 Starting training...")
    print("=" * 50)

    num_epochs = config['training']['num_epochs']
    best_val_acc = 0.0
    save_frequency = config['logging']['save_frequency']

    training_start_time = time.time()

    for epoch in range(num_epochs):
        epoch_start_time = time.time()

        # Train for one epoch
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, config
        )

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Update learning rate
        if scheduler is not None:
            scheduler.step()

        # Calculate epoch duration
        epoch_duration = time.time() - epoch_start_time

        # Print epoch summary
        print(f"\\n📊 Epoch [{epoch+1}/{num_epochs}] Summary:")
        print(
            f"   Training Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%")
        print(f"   Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%")
        print(f"   Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"   Epoch Duration: {epoch_duration:.2f} seconds")

        # Save checkpoint periodically
        if (epoch + 1) % save_frequency == 0:
            checkpoint_filename = config['checkpoint']['checkpoint_filename'].format(
                epoch=epoch+1)
            save_checkpoint(
                model, optimizer, scheduler, epoch+1, val_acc,
                config['checkpoint']['save_dir'], checkpoint_filename
            )

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_best_model(
                model,
                config['checkpoint']['save_dir'],
                config['checkpoint']['best_model_filename'],
                config
            )
            print(
                f"   🎉 New best model! Validation accuracy: {best_val_acc:.2f}%")

    # Training completed
    total_training_time = time.time() - training_start_time
    print("\\n" + "=" * 50)
    print("🎉 Training completed!")
    print(f"Total training time: {total_training_time:.2f} seconds")
    print(".1f")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(
        f"Model saved to: {os.path.join(config['checkpoint']['save_dir'], config['checkpoint']['best_model_filename'])}")
    print("=" * 50)


if __name__ == "__main__":
    main()

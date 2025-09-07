"""
Embryo classification model architecture.

This module provides functions to create and configure deep learning models
for classifying embryo developmental stages.
"""

from typing import Optional
import torch
import torch.nn as nn
from torchvision import models


def create_resnet50_model(num_classes: int = 5, pretrained: bool = True) -> nn.Module:
    """
    Create a ResNet50 model adapted for embryo stage classification.

    This function loads a pre-trained ResNet50 model from torchvision and modifies
    the final classification layer to match the number of embryo stages.

    Args:
        num_classes (int): Number of embryo stages to classify. Defaults to 5
                          (2cell, 4cell, 8cell, blastocyst, morula).
        pretrained (bool): Whether to use pre-trained weights from ImageNet.
                          Defaults to True.

    Returns:
        nn.Module: Configured ResNet50 model ready for training

    Example:
        >>> model = create_resnet50_model(num_classes=5)
        >>> print(model)
    """
    # Load pre-trained ResNet50 model
    model = models.resnet50(
        weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)

    # Freeze early layers to retain pre-trained features
    # Only train the final layers for fine-tuning
    for param in model.parameters():
        param.requires_grad = False

    # Modify the final fully connected layer for our number of classes
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)

    # Unfreeze the final layer for training
    for param in model.fc.parameters():
        param.requires_grad = True

    return model


def get_model_summary(model: nn.Module, input_size: tuple = (3, 224, 224)) -> str:
    """
    Get a summary of the model architecture and parameters.

    Args:
        model (nn.Module): The PyTorch model to summarize
        input_size (tuple): Input tensor size (channels, height, width)

    Returns:
        str: Model summary string
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel()
                           for p in model.parameters() if p.requires_grad)

    summary = ".2e"
    summary += ".2e"
    summary += f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.1f}%)"
    summary += f"Input size: {input_size}"

    return summary


def create_optimizer(model: nn.Module, learning_rate: float = 1e-3) -> torch.optim.Optimizer:
    """
    Create an Adam optimizer for the model.

    Args:
        model (nn.Module): The PyTorch model
        learning_rate (float): Learning rate for the optimizer

    Returns:
        torch.optim.Optimizer: Configured Adam optimizer
    """
    # Only optimize the parameters that require gradients (unfrozen layers)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate
    )
    return optimizer


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    step_size: int = 7,
    gamma: float = 0.1
) -> torch.optim.lr_scheduler.StepLR:
    """
    Create a learning rate scheduler.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer to schedule
        step_size (int): Number of epochs after which to decay learning rate
        gamma (float): Multiplicative factor for learning rate decay

    Returns:
        torch.optim.lr_scheduler.StepLR: Configured learning rate scheduler
    """
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=step_size, gamma=gamma)
    return scheduler


if __name__ == "__main__":
    # Quick test of the model creation
    model = create_resnet50_model(num_classes=5, pretrained=True)
    print("Model created successfully!")
    print(get_model_summary(model))

    # Test optimizer creation
    optimizer = create_optimizer(model, learning_rate=1e-3)
    print(f"Optimizer created: {type(optimizer).__name__}")

    # Test scheduler creation
    scheduler = create_scheduler(optimizer)
    print(f"Scheduler created: {type(scheduler).__name__}")

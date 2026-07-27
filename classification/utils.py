"""
Utility functions: data loading, model construction, reproducibility.
"""

import random
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

import config


def set_seed(seed: int = config.SEED):
    """Set all seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_transforms(train: bool = True):
    """
    Transforms for training (with light data augmentation)
    and for val/test (no augmentation).
    ImageNet normalization since we use transfer learning.
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    if train:
        return transforms.Compose([
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        return transforms.Compose([
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.ToTensor(),
            normalize,
        ])


def get_dataloaders():
    """
    Loads train/val/test from config.TRAIN_DIR / VAL_DIR / TEST_DIR
    using ImageFolder (expected structure: dir/class/*.jpg).
    """
    train_ds = datasets.ImageFolder(config.TRAIN_DIR, transform=get_transforms(train=True))
    val_ds = datasets.ImageFolder(config.VAL_DIR, transform=get_transforms(train=False))
    test_ds = datasets.ImageFolder(config.TEST_DIR, transform=get_transforms(train=False))

    # Safety check: make sure the class order matches config.CLASS_NAMES
    assert train_ds.classes == config.CLASS_NAMES, (
        f"Unexpected class order: {train_ds.classes} != {config.CLASS_NAMES}. "
        f"Adjust config.CLASS_NAMES or rename your folders."
    )

    train_loader = DataLoader(
        train_ds, batch_size=config.BATCH_SIZE, shuffle=True,
        num_workers=config.NUM_WORKERS, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=True,
    )
    return train_loader, val_loader, test_loader


def build_model(model_name: str) -> nn.Module:
    """
    Builds MobileNetV2 or ResNet18 pretrained on ImageNet, with the
    last layer replaced for binary classification.
    """
    if model_name == "mobilenet_v2":
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, config.NUM_CLASSES)

    elif model_name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, config.NUM_CLASSES)

    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model.to(config.DEVICE)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

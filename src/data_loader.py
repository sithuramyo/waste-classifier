import os
import json
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_data_loaders(data_dir, batch_size=32, image_size=224):
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    test_dir = os.path.join(data_dir, 'test')

    # Data augmentation and normalization for training
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Validation and test transforms
    test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Create datasets
    train_dataset = datasets.ImageFolder(train_dir, train_transforms)
    val_dataset = datasets.ImageFolder(val_dir, test_transforms)
    test_dataset = datasets.ImageFolder(test_dir, test_transforms)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Save class indices
    class_to_idx = train_dataset.class_to_idx
    with open(os.path.join('models', 'class_indices.json'), 'w') as f:
        json.dump(class_to_idx, f)

    return train_loader, val_loader, test_loader, class_to_idx
import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class WasteDataset(Dataset):
    def __init__(self, annotation_path, data_root, transform=None):
        self.data_root = data_root
        self.transform = transform
        
        with open(annotation_path) as f:
            data = json.load(f)
        
        # Create mappings
        self.image_id_to_file = {img['image_id']: img['file_name'] for img in data['images']}
        self.category_id_to_name = {cat['id']: cat['name'] for cat in data['categories']}
        
        # Create samples (image_path, category_id)
        self.samples = [
            (self.image_id_to_file[ann['image_id']], self.category_id_to_name[ann['category_id']])
            for ann in data['annotations']
        ]
        
        # Create class mappings
        self.classes = sorted(set(self.category_id_to_name.values()))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_file, category_name = self.samples[idx]
        label = self.class_to_idx[category_name]
        
        # Build path: data_root/category_name/image_file
        img_path = os.path.join(self.data_root, category_name, image_file)
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def get_data_loaders(data_root, batch_size=32, image_size=224):
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = WasteDataset(
        annotation_path=os.path.join(data_root, 'json/train_annotation.json'),
        data_root=os.path.join(data_root, 'train'),
        transform=train_transforms
    )
    
    val_dataset = WasteDataset(
        annotation_path=os.path.join(data_root, 'json/val_annotation.json'),
        data_root=os.path.join(data_root, 'val'),
        transform=test_transforms
    )
    
    test_dataset = WasteDataset(
        annotation_path=os.path.join(data_root, 'json/test_annotation.json'),
        data_root=os.path.join(data_root, 'test'),
        transform=test_transforms
    )

    # Save class indices
    class_indices = {
        'class_to_idx': train_dataset.class_to_idx,
        'idx_to_class': train_dataset.idx_to_class
    }
    os.makedirs('models', exist_ok=True)
    with open('models/class_indices.json', 'w') as f:
        json.dump(class_indices, f, indent=2)

    num_workers = min(4, os.cpu_count() // 2)  # Optimize based on CPU count

    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, persistent_workers=True),
        DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=True),
        DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=True),
        train_dataset.class_to_idx
    )

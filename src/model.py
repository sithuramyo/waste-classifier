import torch
import torch.nn as nn
from torchvision import models
import json

class WasteClassifier(nn.Module):
    def __init__(self, num_classes):
        super(WasteClassifier, self).__init__()
        self.model = models.resnet50(pretrained=True)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)
        
        # Load class indices
        with open('models/class_indices.json', 'r') as f:
            self.class_indices = json.load(f)
        self.idx_to_class = {v: k for k, v in self.class_indices.items()}

    def forward(self, x):
        return self.model(x)
    
    def predict(self, x):
        with torch.no_grad():
            outputs = self.forward(x)
            _, preds = torch.max(outputs, 1)
        return [self.idx_to_class[idx.item()] for idx in preds]

def initialize_model(num_classes, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model = WasteClassifier(num_classes)
    model = model.to(device)
    return model
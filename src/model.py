import torch
import torch.nn as nn
import torchvision.models as models
import json

class WasteClassifier(nn.Module):
    def __init__(self, num_classes):
        super(WasteClassifier, self).__init__()
        self.base_model = models.resnet50(pretrained=True)
        in_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(in_features, num_classes)
        
        # Load class mappings
        with open('models/class_indices.json', 'r') as f:
            class_indices = json.load(f)
        self.idx_to_class = class_indices['idx_to_class']

    def forward(self, x):
        return self.base_model(x)
    
    def predict(self, x):
        with torch.no_grad():
            outputs = self.forward(x)
            _, preds = torch.max(outputs, 1)
        return [self.idx_to_class[str(idx.item())] for idx in preds]

def initialize_model(num_classes, feature_extract, use_pretrained=True):
    model_ft = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if use_pretrained else None)
    # Modify the model as needed
    # ...
    return model_ft
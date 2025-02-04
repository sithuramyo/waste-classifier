import torch
import torch.nn as nn
import torchvision.models as models
import json

class WasteClassifier(nn.Module):
    def __init__(self, num_classes, load_classes=True):
        super(WasteClassifier, self).__init__()
        
        # Load pretrained ResNet50
        self.base_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        in_features = self.base_model.fc.in_features
        
        # Replace final layer with custom classifier
        self.base_model.fc = nn.Linear(in_features, num_classes)
        
        # Load class mappings (Optional)
        self.idx_to_class = None
        if load_classes:
            try:
                with open('models/class_indices.json', 'r') as f:
                    class_indices = json.load(f)
                self.idx_to_class = {int(k): v for k, v in class_indices['idx_to_class'].items()}
            except FileNotFoundError:
                print("Warning: class_indices.json not found. Predictions will return class indices.")

    def forward(self, x):
        return self.base_model(x)
    
    def predict(self, x):
        """ Returns class names for given input tensor """
        with torch.no_grad():
            outputs = self.forward(x)
            _, preds = torch.max(outputs, 1)
        
        # Return class names if available, else return indices
        if self.idx_to_class:
            return [self.idx_to_class[idx.item()] for idx in preds]
        return preds.tolist()

def initialize_model(num_classes, feature_extract=True, use_pretrained=True):
    """ Initialize the ResNet50 model with an updated classifier layer """
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if use_pretrained else None)
    
    # Freeze parameters if feature extraction is enabled
    if feature_extract:
        for param in model.parameters():
            param.requires_grad = False
    
    # Replace final FC layer for classification
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    
    return model

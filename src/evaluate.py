import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os
from model import initialize_model
from data_loader import get_data_loaders

def evaluate_model(model_path, data_dir, batch_size=32):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data
    _, _, test_loader, class_to_idx = get_data_loaders(data_dir, batch_size)
    num_classes = len(class_to_idx)
    
    # Load model
    model = initialize_model(num_classes, feature_extract=True, use_pretrained=True)
    model = model.to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    
    # Evaluate
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Generate classification report and confusion matrix
    print(classification_report(all_labels, all_preds, target_names=class_to_idx.keys()))
    cm = confusion_matrix(all_labels, all_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_to_idx.keys(), yticklabels=class_to_idx.keys())
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
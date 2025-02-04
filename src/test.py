import torch
import os
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from model import WasteClassifier  # Import your model

# Load the trained model
def load_model(checkpoint_path='models/model_best.pth.tar', num_classes=5, device=None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Initialize the model
    model = WasteClassifier(num_classes)
    model.to(device)

    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()  # Set model to evaluation mode

    print(f"Model loaded from {checkpoint_path} on {device}")
    return model, device

# Preprocess the image
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to model input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)  # Add batch dimension

# Run inference on a single image
def test_image(model, image_path, device):
    if not os.path.exists(image_path):
        print(f"Error: Image file not found: {image_path}")
        return None

    image_tensor = preprocess_image(image_path).to(device)
    prediction = model.predict(image_tensor, device=device)
    
    return prediction[0] if isinstance(prediction, list) else prediction  # Return class name or index

# Main testing function
if __name__ == '__main__':
    num_classes = 5  # Change this based on your dataset
    model, device = load_model(num_classes=num_classes)

    test_image_path = 'tests/images.jpg'
    prediction = test_image(model, test_image_path, device)

    if prediction is not None:
        print(f"Predicted class: {prediction}")

        # Display the image with prediction
        image = Image.open(test_image_path)
        plt.imshow(image)
        plt.title(f"Predicted Class: {prediction}")
        plt.axis('off')
        plt.show()

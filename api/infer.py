import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from model import Net
# Define the same network structure

# Load the saved model
def load_model(path="global_model.pth"):
    model = Net()
    try:
        model.load_state_dict(torch.load(path))
        print(f"Model loaded successfully from {path}")
    except Exception as e:
        print(f"Error loading model: {e}")
    model.eval()
    return model

# Preprocess the input image
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Grayscale(),         # Convert to grayscale
        transforms.Resize((28, 28)),    # Resize to 28x28 pixels
        transforms.ToTensor(),          # Convert to tensor
        transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
    ])
    try:
        image = Image.open(image_path).convert("L")
        print(f"Image loaded successfully: {image_path}")
        return transform(image).unsqueeze(0)
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

# Predict the digit
def predict(image_path, model):
    input_tensor = preprocess_image(image_path)
    if input_tensor is None:
        return
    print(f"Input Tensor Shape: {input_tensor.shape}")
    
    with torch.no_grad():
        output = model(input_tensor)
        print(f"Model Output: {output}")
        _, predicted = torch.max(output, 1)
        print(f"Predicted Digit: {predicted.item()}")

# Test the model
if __name__ == "__main__":
    image_path = "7.png"  # Replace with the path to your test image
    model = load_model("global_model.pth")  # Load the trained model
    predict(image_path, model)

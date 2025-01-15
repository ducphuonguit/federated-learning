from flask import Flask, request, jsonify
import torch
from model import Net  # Ensure your model is defined in model.py
from PIL import Image
import io
import torchvision.transforms as transforms
from flask_cors import CORS

# Initialize the Flask app
app = Flask(__name__)
CORS(app) 
# Define and load the model
def load_model(path="global_model.pth"):
    model = Net()
    try:
        model.load_state_dict(torch.load(path))
        print(f"Model loaded successfully from {path}")
    except Exception as e:
        print(f"Error loading model: {e}")
    model.eval()
    return model

model = load_model()

# Preprocess the input image
def preprocess_image(image_bytes):
    transform = transforms.Compose([
        transforms.Grayscale(),         # Convert to grayscale
        transforms.Resize((28, 28)),    # Resize to 28x28 pixels
        transforms.ToTensor(),          # Convert to tensor
        transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
    ])
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("L")
        return transform(image).unsqueeze(0)
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

# Define the predict function
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image_file = request.files['image']
    try:
        image_bytes = image_file.read()
        input_tensor = preprocess_image(image_bytes)
        if input_tensor is None:
            return jsonify({'error': 'Invalid image'}), 400
        
        with torch.no_grad():
            output = model(input_tensor)
            _, predicted = torch.max(output, 1)
            print("Predicted digit:", predicted.item())
            return jsonify({'predicted': predicted.item()})
    except Exception as e:
        print(f"Error during prediction: {e}")
        # return jsonify({'error': 'Failed to process the image'}), 500

# Run the app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8081)

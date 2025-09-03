from flask import Flask, request, jsonify
from flask_cors import CORS
from torchvision import models, transforms
from PIL import Image
import torch
import requests
import io

app = Flask(__name__)
CORS(app)

# Load ResNet50 model with pretrained weights
from torchvision.models import resnet50, ResNet50_Weights
weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=weights)
model.eval()

# Image transformation pipeline
transform = weights.transforms()

# Load ImageNet class labels
LABELS_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
classes = requests.get(LABELS_URL).text.strip().split("\n")

@app.route('/predict', methods=['POST'])
def predict():
    # Expect JSON payload with image_url
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    image_url = data.get('image_url')
    if not image_url:
        return jsonify({"error": "No image_url provided"}), 400

    try:
        # Download image from URL
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()
        img = Image.open(io.BytesIO(response.content)).convert('RGB')
        img_t = transform(img).unsqueeze(0)
    except Exception as e:
        return jsonify({"error": f"Failed to process image: {str(e)}"}), 500

    try:
        # Perform prediction
        with torch.no_grad():
            outputs = model(img_t)
            _, idx = torch.max(outputs, 1)
            label = classes[idx.item()]
            confidence = torch.softmax(outputs, dim=1)[0, idx].item()

        return jsonify({
            "prediction": label,
            "confidence": round(float(confidence), 4)
        })
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
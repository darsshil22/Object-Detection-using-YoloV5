import os
from flask import Flask, request, jsonify
from ultralytics import YOLO
from PIL import Image

# Initialize Flask app
app = Flask(__name__)

# Path to the trained YOLO model
MODEL_PATH = "D:/AIDI/Object Detection Project/models/checkpoints/yolo_training/weights/yolov5s.pt"
model = YOLO(MODEL_PATH)  # Load the YOLO model

# Root route
@app.route('/')
def index():
    return jsonify({'message': 'Welcome to the YOLO model API! Use /predict to make predictions.'})

# Define the endpoint for predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the image from the request
        image_file = request.files.get('image')
        if image_file is None:
            return jsonify({'error': 'No image provided'}), 400
        
        # Convert image to a format YOLO can work with
        image = Image.open(image_file.stream)

        # Perform inference with the YOLO model
        results = model(image)

        # Extract predictions
        predictions = results.pandas().xywh[0].to_dict(orient='records')  # Convert to a list of dictionaries

        return jsonify({'predictions': predictions})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Endpoint to check if the server is running
@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'Model server is running'}), 200

# Run the Flask app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

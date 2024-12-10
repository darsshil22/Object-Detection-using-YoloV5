import os
import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt

# Paths
YOLO_MODEL = "D:/AIDI/Object Detection Project/models/checkpoints/yolo_training/weights/yolov5s.pt"  # Replace with your trained model path
IMAGE_PATH = "D:/AIDI/Object Detection Project/data/processed/test/images/000000027519.jpg"  # Replace with your test image path

# Initialize YOLO model
model = YOLO(YOLO_MODEL)

def draw_predictions(image_path, predictions):
    """
    Draw bounding boxes and labels on the image.
    """
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    for pred in predictions:
        x1, y1, x2, y2, conf, class_id = pred[:6]
        label = f"{model.names[int(class_id)]} {conf:.2f}"

        # Draw bounding box
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        # Put label text
        cv2.putText(image, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    return image

def main():
    # Run inference on a single image
    results = model.predict(IMAGE_PATH, conf=0.25)

    # Extract predictions (bounding boxes, confidence, and class IDs)
    predictions = results[0].boxes.data.cpu().numpy()  # YOLOv5-style output

    # Draw predictions on the image
    annotated_image = draw_predictions(IMAGE_PATH, predictions)

    # Display the image
    plt.figure(figsize=(10, 10))
    plt.imshow(annotated_image)
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()

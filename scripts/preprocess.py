import os
from ultralytics import YOLO

# Paths
PROCESSED_TRAIN_DIR = "D:/AIDI/Object Detection Project/data/processed/train"
PROCESSED_VAL_DIR = "D:/AIDI/Object Detection Project/data/processed/val"
YOLO_DATA_CONFIG = "D:/AIDI/Object Detection Project/data/yolo_data.yaml"
YOLO_MODEL = "yolov5s.pt"  # Pretrained YOLO model (e.g., yolov5s.pt)
OUTPUT_DIR = "D:/AIDI/Object Detection Project/models/checkpoints"

# Training parameters
# Training parameters
EPOCHS = 20  
BATCH_SIZE = 8 
IMG_SIZE = 416 


# Generate YOLO data configuration
def create_yolo_data_config():
    config_content = f"""
train: {PROCESSED_TRAIN_DIR}/images
val: {PROCESSED_VAL_DIR}/images

nc: 5  # Number of classes
names: ["person", "bicycle", "car", "dog", "chair"]
    """
    with open(YOLO_DATA_CONFIG, "w") as f:
        f.write(config_content)

def main():
    # Ensure YOLO data config exists
    if not os.path.exists(YOLO_DATA_CONFIG):
        print("Creating YOLO data configuration...")
        create_yolo_data_config()
        print(f"YOLO data configuration saved at: {YOLO_DATA_CONFIG}")

    # Initialize YOLO model
    print("Initializing YOLO model...")
    model = YOLO(YOLO_MODEL)

    # Train the model
    print("Starting training...")
    model.train(
        data=YOLO_DATA_CONFIG,
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        imgsz=IMG_SIZE,
        project=OUTPUT_DIR,
        name="yolo_training",
        save_period=5,
        workers=4,
    )

    print(f"Training completed. Results saved in: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()

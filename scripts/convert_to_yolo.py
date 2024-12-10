import os
import json
import shutil
import cv2
from pycocotools.coco import COCO

# Paths
PROCESSED_DATA_DIR = "D:/AIDI/Object Detection Project/data/processed"
TRAIN_IMAGES_DIR = os.path.join(PROCESSED_DATA_DIR, "train/images")
VAL_IMAGES_DIR = os.path.join(PROCESSED_DATA_DIR, "val/images")
TRAIN_LABELS_DIR = os.path.join(PROCESSED_DATA_DIR, "train/labels")
VAL_LABELS_DIR = os.path.join(PROCESSED_DATA_DIR, "val/labels")

# Ensure label directories exist
os.makedirs(TRAIN_LABELS_DIR, exist_ok=True)
os.makedirs(VAL_LABELS_DIR, exist_ok=True)

# Load annotations
TRAIN_ANNOTATIONS_FILE = os.path.join(PROCESSED_DATA_DIR, 'train/annotations.json')
VAL_ANNOTATIONS_FILE = os.path.join(PROCESSED_DATA_DIR, 'val/annotations.json')

with open(TRAIN_ANNOTATIONS_FILE, 'r') as f:
    train_annotations = json.load(f)
with open(VAL_ANNOTATIONS_FILE, 'r') as f:
    val_annotations = json.load(f)

# Combine annotations
all_annotations = train_annotations['annotations'] + val_annotations['annotations']

# Create a function to convert COCO to YOLO format
def coco_to_yolo_format(img_info, annotations, images_dir, labels_dir, class_names):
    img_path = os.path.join(images_dir, img_info['file_name'])
    img = cv2.imread(img_path)
    h, w, _ = img.shape
    
    label_file = os.path.join(labels_dir, img_info['file_name'].replace('.jpg', '.txt'))
    
    with open(label_file, 'w') as f:
        for ann in annotations:
            if ann['image_id'] == img_info['id']:
                category_id = ann['category_id'] - 1  # Adjust to 0-index (subtract 1)
                if category_id < len(class_names):
                    class_id = category_id
                else:
                    print(f"Warning: Category ID {category_id + 1} not found in class names.")
                    continue
                
                # Convert bbox to YOLO format (normalize)
                bbox = ann['bbox']
                x_center = (bbox[0] + bbox[2] / 2) / w
                y_center = (bbox[1] + bbox[3] / 2) / h
                width = bbox[2] / w
                height = bbox[3] / h
                
                # Write YOLO format line
                f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

# List your selected class names (the ones in the dataset)
class_names = ['person', 'bicycle', 'car', 'dog', 'chair']

# Process train annotations
for img_info in train_annotations['images']:
    annotations = [ann for ann in all_annotations if ann['image_id'] == img_info['id']]
    coco_to_yolo_format(img_info, annotations, TRAIN_IMAGES_DIR, TRAIN_LABELS_DIR, class_names)

# Process validation annotations
for img_info in val_annotations['images']:
    annotations = [ann for ann in all_annotations if ann['image_id'] == img_info['id']]
    coco_to_yolo_format(img_info, annotations, VAL_IMAGES_DIR, VAL_LABELS_DIR, class_names)

print("Labels generated and saved in YOLO format.")

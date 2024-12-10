import os
import json
import random
import shutil

# Paths
PROCESSED_DATA_DIR = "D:/AIDI/Object Detection Project/data/processed"
TRAIN_DIR = os.path.join(PROCESSED_DATA_DIR, 'train')
VAL_DIR = os.path.join(PROCESSED_DATA_DIR, 'val')
TEST_DIR = os.path.join(PROCESSED_DATA_DIR, 'test')

# Ensure the test directory exists
os.makedirs(TEST_DIR, exist_ok=True)
os.makedirs(os.path.join(TEST_DIR, 'images'), exist_ok=True)

# Load train and val annotations
TRAIN_ANNOTATIONS_FILE = os.path.join(TRAIN_DIR, 'annotations.json')
VAL_ANNOTATIONS_FILE = os.path.join(VAL_DIR, 'annotations.json')

if not os.path.exists(TRAIN_ANNOTATIONS_FILE) or not os.path.exists(VAL_ANNOTATIONS_FILE):
    raise FileNotFoundError("Annotations not found in train or val directories.")

# Load annotations
with open(TRAIN_ANNOTATIONS_FILE, 'r') as f:
    train_annotations = json.load(f)
with open(VAL_ANNOTATIONS_FILE, 'r') as f:
    val_annotations = json.load(f)

# Combine train and val annotations
all_images = train_annotations["images"] + val_annotations["images"]
all_annotations = train_annotations["annotations"] + val_annotations["annotations"]

# Shuffle images for splitting
random.shuffle(all_images)

# Split sizes
train_size = int(0.8 * len(all_images))
val_size = int(0.1 * len(all_images))
test_size = len(all_images) - train_size - val_size

# Split images
train_images = all_images[:train_size]
val_images = all_images[train_size:train_size + val_size]
test_images = all_images[train_size + val_size:]

# Helper: Get image IDs for each split
def get_image_ids(image_list):
    return {img["id"] for img in image_list}

train_ids = get_image_ids(train_images)
val_ids = get_image_ids(val_images)
test_ids = get_image_ids(test_images)

# Filter annotations based on image IDs
def filter_annotations(image_ids, annotations):
    return [ann for ann in annotations if ann["image_id"] in image_ids]

train_annotations_filtered = filter_annotations(train_ids, all_annotations)
val_annotations_filtered = filter_annotations(val_ids, all_annotations)
test_annotations_filtered = filter_annotations(test_ids, all_annotations)

def save_split(images, annotations, output_dir):
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    for img in images:
        src_path = os.path.join(PROCESSED_DATA_DIR, 'train', 'images', img["file_name"])
        dst_path = os.path.join(output_dir, 'images', img["file_name"])
        
        # Copy only if the source and destination are different
        if os.path.abspath(src_path) != os.path.abspath(dst_path):
            shutil.copy(src_path, dst_path)

    with open(os.path.join(output_dir, 'annotations.json'), 'w') as f:
        json.dump({"images": images, "annotations": annotations}, f)


save_split(train_images, train_annotations_filtered, TRAIN_DIR)
save_split(val_images, val_annotations_filtered, VAL_DIR)
save_split(test_images, test_annotations_filtered, TEST_DIR)

print("Data successfully split into train, val, and test sets.")

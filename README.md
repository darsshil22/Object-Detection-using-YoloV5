# Object Detection Project with YOLO

This project demonstrates object detection using the YOLO model trained on the COCO dataset. The workflow includes preprocessing, training, and evaluation of the model.


## Dataset

### Download COCO Dataset

The dataset used in this project is from the COCO dataset. You need to download the following files from [COCO Dataset](https://cocodataset.org/#download):

- **2017 Train images [118K/18GB]**
- **2017 Validation images [5K/1GB]**
- **2017 Test images**
- **2017 Annotations [241MB]**

After downloading:
- Place the train, val, and test images in the `data/raw/` directory.
- Place the `instances_train2017.json` and `instances_val2017.json` files in the `data/raw/annotations/` directory.

---

## Project Structure

```
Object Detection Project/
├── data/
│   ├── raw/
│   │   ├── train2017/               # Raw training images
│   │   ├── val2017/                 # Raw validation images
│   │   ├── test2017/                # Raw test images
│   │   └── annotations/             # COCO annotation files
│   ├── processed/                   # Preprocessed dataset
│       ├── train/                   # Processed training data
│       ├── val/                     # Processed validation data
│       └── test/                    # Processed test data
├── models/
│   └── checkpoints/                 # YOLO training checkpoints
├── scripts/
│   ├── preprocess.py                # Data preprocessing script
│   ├── convert_to_yolo.py           # Conversion to YOLO format
│   ├── train.py                     # Model training script
│   ├── test.py                      # Test a single image
│   └── evaluate.py                  # Evaluate the model
├── requirements.txt                 # Dependencies
└── README.md                        # Project overview
```

---

## Usage

### 1. Preprocess the Data
Run the script to resize images and filter annotations:
```bash
python scripts/preprocess.py
```

### 2. Convert Annotations to YOLO Format
Convert the COCO annotations to YOLO format:
```bash
python scripts/convert_to_yolo.py
```

### 3. Train the Model
Train the YOLO model using the preprocessed dataset:
```bash
python scripts/train.py
```

### 4. Test a Single Image
Run inference on a single image:
```bash
python scripts/test.py
```

### 5. Evaluate the Model
Evaluate the model's performance on validation and test sets:
```bash
python scripts/evaluate.py
```

---

## Results

Results will be saved in the `models/checkpoints/` directory. Example metrics include precision, recall, and F1-score.

---

## Acknowledgments

This project uses the [YOLO](https://github.com/ultralytics/yolov5) framework and the [COCO Dataset](https://cocodataset.org/#home).

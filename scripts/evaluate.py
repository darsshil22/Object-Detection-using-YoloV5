import os
import json
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from ultralytics import YOLO

# Paths
PROCESSED_DATA_DIR = "D:/AIDI/Object Detection Project/data/processed"
TRAIN_DIR = os.path.join(PROCESSED_DATA_DIR, 'train')
VAL_DIR = os.path.join(PROCESSED_DATA_DIR, 'val')
TEST_DIR = os.path.join(PROCESSED_DATA_DIR, 'test')
YOLO_MODEL = "D:\AIDI\Object Detection Project\models\checkpoints\yolo_training\weights/yolov5s.pt"  # Replace with your trained model path

# Helper function to evaluate metrics
def evaluate_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted")
    recall = recall_score(y_true, y_pred, average="weighted")
    f1 = f1_score(y_true, y_pred, average="weighted")
    return accuracy, precision, recall, f1

# Function to extract true labels and predicted labels for evaluation
def get_true_and_predicted_labels(annotations_file, predictions):
    with open(annotations_file, 'r') as f:
        annotations = json.load(f)
    
    y_true = []  # Ground truth
    y_pred = []  # Model predictions

    # Extract the relevant information from annotations and predictions
    for annotation in annotations['annotations']:
        y_true.append(annotation['category_id'])  # Assuming category_id is the label
        
    for pred in predictions:
        y_pred.append(pred['category_id'])  # Assuming category_id is the predicted label

    return y_true, y_pred

# Initialize YOLO model
model = YOLO(YOLO_MODEL)

# Evaluate on Validation Set
print("Evaluating on Validation Set...")
val_predictions = model.predict(VAL_DIR + '/images')
val_results = val_predictions.pandas().xywh  # Assuming output is in pandas DataFrame format

y_true_val, y_pred_val = get_true_and_predicted_labels(os.path.join(VAL_DIR, 'annotations.json'), val_results)
accuracy_val, precision_val, recall_val, f1_val = evaluate_metrics(y_true_val, y_pred_val)

print(f"Validation Set Metrics - Accuracy: {accuracy_val}, Precision: {precision_val}, Recall: {recall_val}, F1: {f1_val}")

# Evaluate on Test Set
print("Evaluating on Test Set...")
test_predictions = model.predict(TEST_DIR + '/images')
test_results = test_predictions.pandas().xywh  # Assuming output is in pandas DataFrame format

y_true_test, y_pred_test = get_true_and_predicted_labels(os.path.join(TEST_DIR, 'annotations.json'), test_results)
accuracy_test, precision_test, recall_test, f1_test = evaluate_metrics(y_true_test, y_pred_test)

print(f"Test Set Metrics - Accuracy: {accuracy_test}, Precision: {precision_test}, Recall: {recall_test}, F1: {f1_test}")

import os
import sys
import yaml
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from ultralytics import YOLO
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd

def load_config():
    """Load the YOLO configuration file."""
    config_path = Path(__file__).parent.parent / "models" / "yolo_params.yaml"
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def evaluate_model():
    """Evaluate the trained YOLOv8 model on the test set."""
    
    # Get the project directory
    project_dir = Path(__file__).parent.parent
    os.chdir(project_dir)
    
    # Load configuration
    config = load_config()
    class_names = config['names']
    num_classes = config['nc']
    
    print("=" * 60)
    print("YOLOv8 Model Evaluation")
    print("=" * 60)
    print(f"Classes: {class_names}")
    print(f"Number of classes: {num_classes}")
    print()
    
    # Load the trained model
    model_path = project_dir / "runs" / "detect" / "train" / "weights" / "best.pt"
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        print("Please ensure training has been completed successfully.")
        return
    
    print(f"Loading model from: {model_path}")
    model = YOLO(model_path)
    
    # Get test dataset path
    test_path = project_dir / "datasets" / "space_station" / "test" / "images"
    if not test_path.exists():
        print(f"Error: Test dataset not found at {test_path}")
        return
    
    print(f"Test dataset path: {test_path}")
    print()
    
    # Run validation on test set
    print("Running evaluation on test set...")
    results = model.val(
        data=project_dir / "models" / "yolo_params.yaml",
        split="test",
        verbose=True
    )
    
    # Extract metrics
    metrics = results.results_dict
    
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    
    # Print overall metrics
    mAP50 = metrics.get('metrics/mAP50', 'N/A')
    mAP50_95 = metrics.get('metrics/mAP50-95', 'N/A')
    precision = metrics.get('metrics/precision', 'N/A')
    recall = metrics.get('metrics/recall', 'N/A')
    
    print(f"mAP@0.5: {mAP50:.4f}" if isinstance(mAP50, (int, float)) else f"mAP@0.5: {mAP50}")
    print(f"mAP@0.5:0.95: {mAP50_95:.4f}" if isinstance(mAP50_95, (int, float)) else f"mAP@0.5:0.95: {mAP50_95}")
    print(f"Precision: {precision:.4f}" if isinstance(precision, (int, float)) else f"Precision: {precision}")
    print(f"Recall: {recall:.4f}" if isinstance(recall, (int, float)) else f"Recall: {recall}")
    
    # Print per-class metrics if available
    if hasattr(results, 'names') and results.names:
        print("\nPer-Class Metrics:")
        print("-" * 40)
        for i, class_name in enumerate(results.names):
            if f'metrics/mAP50(B)' in metrics:
                mAP50_data = metrics.get(f'metrics/mAP50(B)', [])
                if isinstance(mAP50_data, (list, tuple)) and i < len(mAP50_data):
                    mAP50 = mAP50_data[i]
                    print(f"{class_name}: mAP@0.5 = {mAP50:.4f}")
                elif isinstance(mAP50_data, (int, float)):
                    # If it's a single value, use it for all classes
                    print(f"{class_name}: mAP@0.5 = {mAP50_data:.4f}")
                else:
                    print(f"{class_name}: mAP@0.5 = N/A")
    
    # Generate confusion matrix
    print("\n" + "=" * 60)
    print("GENERATING CONFUSION MATRIX")
    print("=" * 60)
    
    # Run predictions on test set to get confusion matrix
    test_images = list(test_path.glob("*.jpg")) + list(test_path.glob("*.png"))
    if not test_images:
        print("No test images found!")
        return
    
    print(f"Found {len(test_images)} test images")
    
    # Collect predictions and ground truth
    all_predictions = []
    all_ground_truth = []
    
    for img_path in test_images:
        # Get predictions
        pred_results = model.predict(img_path, conf=0.25, verbose=False)
        
        # Get ground truth labels
        label_path = img_path.parent.parent / "labels" / f"{img_path.stem}.txt"
        
        # Process predictions
        for pred in pred_results:
            if pred.boxes is not None:
                for box in pred.boxes:
                    if box.conf > 0.25:  # Confidence threshold
                        cls_id = int(box.cls[0])
                        all_predictions.append(cls_id)
        
        # Process ground truth
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        gt_cls_id = int(parts[0])
                        all_ground_truth.append(gt_cls_id)
    
    # Create confusion matrix
    if all_predictions and all_ground_truth:
        # Pad with zeros to make equal length
        max_len = max(len(all_predictions), len(all_ground_truth))
        all_predictions.extend([0] * (max_len - len(all_predictions)))
        all_ground_truth.extend([0] * (max_len - len(all_ground_truth)))
        
        cm = confusion_matrix(all_ground_truth, all_predictions, labels=range(num_classes))
        
        print("\nConfusion Matrix:")
        print("-" * 40)
        
        # Create DataFrame for better display
        cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
        print(cm_df)
        
        # Calculate additional metrics
        print("\nClassification Report:")
        print("-" * 40)
        print(classification_report(all_ground_truth, all_predictions, 
                                  target_names=class_names, zero_division=0))
        
        # Save confusion matrix plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix - YOLOv8 Model Evaluation')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        
        # Save the plot
        plot_path = project_dir / "confusion_matrix.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"\nConfusion matrix plot saved to: {plot_path}")
        plt.close()
    
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    evaluate_model() 
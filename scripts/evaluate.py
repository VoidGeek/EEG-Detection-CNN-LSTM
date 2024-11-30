import os
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, auc
from keras import models

def evaluate_model(model_path, features_file, labels_file):
    print("[INFO] Loading preprocessed data and model.")
    
    if not os.path.exists(features_file) or not os.path.exists(labels_file):
        raise FileNotFoundError("[ERROR] Preprocessed data not found. Please run the preprocessing script first.")
    if not os.path.exists(model_path):
        raise FileNotFoundError("[ERROR] Model file not found. Please train the model first.")

    X_test = np.load(features_file)
    y_test = np.load(labels_file)

    print("[INFO] Reshaping data for evaluation.")
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # No need to apply np.argmax if y_test is already a 1D array of labels
    if y_test.ndim == 2:  # Check if y_test is one-hot encoded
        y_test = np.argmax(y_test, axis=1)

    print("[INFO] Loading model.")
    model = models.load_model(model_path)

    print("[INFO] Predicting on test data.")
    y_pred_probs = model.predict(X_test)
    y_pred = (y_pred_probs[:, 1] > 0.5).astype(int)

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Non-Seizure", "Seizure"], zero_division=0))

    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_probs[:, 1])
    pr_auc = auc(recall, precision)
    print(f"\nPrecision-Recall AUC: {pr_auc:.4f}")

if __name__ == "__main__":
    model_path = "../model/model.h5"
    features_file = "../processed_data/features.npy"
    labels_file = "../processed_data/labels.npy"

    print("[INFO] Starting evaluation process.")
    evaluate_model(model_path, features_file, labels_file)
    print("[INFO] Evaluation completed successfully.")

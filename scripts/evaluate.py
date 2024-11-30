import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, auc
from keras import models

def evaluate_model(model_path, features_file, labels_file):
    print("[INFO] Loading preprocessed data and model.")
    X_test = np.load(features_file)
    y_test = np.load(labels_file)

    print("[INFO] Reshaping data for evaluation.")
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    y_test = np.argmax(y_test, axis=1)

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

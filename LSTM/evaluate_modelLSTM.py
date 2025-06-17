import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    roc_curve, 
    auc, 
    precision_recall_curve, 
    average_precision_score
)
from keras.models import load_model

def comprehensive_model_evaluation(model_path, X_test_path, y_test_path):
    """
    Comprehensive model evaluation with multiple performance metrics
    
    Args:
        model_path (str): Path to saved Keras model
        X_test_path (str): Path to test features
        y_test_path (str): Path to test labels
    """
    # 1. Load Model and Data
    model = load_model(model_path)
    X_test = np.load(X_test_path)
    y_test = np.load(y_test_path)

    # 2. Preprocess Data
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    X_test = (X_test - X_test.mean(axis=0)) / X_test.std(axis=0)
    X_test = np.nan_to_num(X_test)

    # 3. Model Evaluation
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=2)
    print(f"Test Loss: {test_loss}")
    print(f"Test Accuracy: {test_accuracy}")

    # 4. Predictions
    y_test_pred_probs = model.predict(X_test).flatten()
    y_test_pred = (y_test_pred_probs > 0.5).astype(int)

    # 5. Classification Report
    print("\nClassification Report:")
    print(classification_report(y_test, y_test_pred, target_names=["No Seizure", "Seizure"]))

    # 6. Confusion Matrix
    conf_matrix_test = confusion_matrix(y_test, y_test_pred)
    print("\nTest Confusion Matrix:")
    print(conf_matrix_test)

    # 7. Performance Visualization Methods
    visualize_confusion_matrix(conf_matrix_test)
    plot_roc_curve(y_test, y_test_pred_probs)
    plot_precision_recall_curve(y_test, y_test_pred_probs)
    
    # 8. Advanced Performance Metrics
    calculate_additional_metrics(y_test, y_test_pred, y_test_pred_probs)

def visualize_confusion_matrix(conf_matrix):
    """
    Visualize Confusion Matrix
    
    Args:
        conf_matrix (np.array): Confusion matrix
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        conf_matrix, 
        annot=True, 
        fmt='d', 
        cmap='Blues', 
        xticklabels=["No Seizure", "Seizure"], 
        yticklabels=["No Seizure", "Seizure"]
    )
    plt.title("Seizure Detection Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.show()

def plot_roc_curve(y_true, y_pred_proba):
    """
    Plot ROC Curve
    
    Args:
        y_true (np.array): True labels
        y_pred_proba (np.array): Predicted probabilities
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(
        fpr, tpr, 
        color='darkorange', 
        lw=2, 
        label=f'ROC curve (AUC = {roc_auc:.2f})'
    )
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig('roc_curve.png')
    plt.show()

def plot_precision_recall_curve(y_true, y_pred_proba):
    """
    Plot Precision-Recall Curve
    
    Args:
        y_true (np.array): True labels
        y_pred_proba (np.array): Predicted probabilities
    """
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    avg_precision = average_precision_score(y_true, y_pred_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(
        recall, precision, 
        color='blue', 
        label=f'Precision-Recall curve (AP = {avg_precision:.2f})'
    )
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig('precision_recall_curve.png')
    plt.show()

def calculate_additional_metrics(y_true, y_pred, y_pred_proba):
    """
    Calculate Additional Performance Metrics
    
    Args:
        y_true (np.array): True labels
        y_pred (np.array): Predicted labels
        y_pred_proba (np.array): Predicted probabilities
    """
    # Compute Confusion Matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Performance Metrics
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    precision = tp / (tp + fp)
    f1_score = 2 * (precision * sensitivity) / (precision + sensitivity)

    print("\nDetailed Performance Metrics:")
    print(f"True Positives: {tp}")
    print(f"True Negatives: {tn}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print(f"Sensitivity: {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"F1 Score: {f1_score:.4f}")

# Main Execution
if __name__ == "__main__":
    comprehensive_model_evaluation(
        model_path='epileptic_seizure_detection_lstm_model.h5',
        X_test_path='E:\EEG-Detection-CNN-LSTM\X_test1.npy',
        y_test_path='E:\EEG-Detection-CNN-LSTM\y_test1.npy'
    )
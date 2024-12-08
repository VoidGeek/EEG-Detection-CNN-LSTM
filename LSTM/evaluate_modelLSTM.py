import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from keras.models import load_model
import matplotlib.pyplot as plt
import seaborn as sns

# Load the saved model
model = load_model('epileptic_seizure_detection_lstm_model.h5')

# Load the saved test set
X_test = np.load('X_test1.npy')
y_test = np.load('y_test1.npy')

# Reshape X_test to 3D for LSTM
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Normalize input data
X_test = (X_test - X_test.mean(axis=0)) / X_test.std(axis=0)
X_test = np.nan_to_num(X_test)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=2)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

# Predictions for the test set
y_test_pred_probs = model.predict(X_test)
y_test_pred = (y_test_pred_probs > 0.5).astype(int).flatten()

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_test_pred, target_names=["No Seizure", "Seizure"]))

# Confusion matrix
conf_matrix_test = confusion_matrix(y_test, y_test_pred)
print("\nTest Confusion Matrix:")
print(conf_matrix_test)

plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix_test, annot=True, fmt='d', cmap='Blues', xticklabels=["No Seizure", "Seizure"], yticklabels=["No Seizure", "Seizure"])
plt.title("Test Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

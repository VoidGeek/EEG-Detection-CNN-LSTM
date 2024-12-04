import numpy as np
import pandas as pd
import random
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from keras.models import load_model

# Parameters
n_features = 178  # Number of original features
n_samples = 500  # Total samples to generate
seizure_proportion = 0.5  # Proportion of seizure samples

# Generate synthetic EEG signals
data = []
labels = []
n_seizure = int(seizure_proportion * n_samples)
n_non_seizure = n_samples - n_seizure

# Signal generation for Seizure and Non-Seizure categories
for _ in range(n_seizure):
    signal = np.random.uniform(-1, 1, n_features) * np.sin(np.linspace(0, 20 * np.pi, n_features)) * random.uniform(100, 300)
    signal += np.random.normal(0, 10, n_features)  # Add noise
    data.append(signal)
    labels.append(1)  # Seizure

for _ in range(n_non_seizure):
    signal = np.random.uniform(-1, 1, n_features) * np.sin(np.linspace(0, 10 * np.pi, n_features)) * random.uniform(20, 80)
    signal += np.random.normal(0, 5, n_features)  # Add noise
    data.append(signal)
    labels.append(0)  # Non-Seizure

# Convert to NumPy arrays
data = np.array(data)
labels = np.array(labels)

# Shuffle the data and labels together
indices = np.arange(n_samples)
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]

# Visualize the first 5 seizure and non-seizure signals separately
plt.figure(figsize=(15, 10))

# Plot first 5 Seizure signals
for i in range(5):
    plt.subplot(5, 2, 2*i + 1)
    plt.plot(data[labels == 1][i], label=f"Seizure Signal {i+1}")
    plt.title(f"Seizure EEG Signal {i+1}")
    plt.xlabel("Time (arbitrary units)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.ylim([-250, 250])  # Set y-axis limits to be the same for all plots

# Plot first 5 Non-Seizure signals
for i in range(5):
    plt.subplot(5, 2, 2*i + 2)
    plt.plot(data[labels == 0][i], label=f"Non-Seizure Signal {i+1}")
    plt.title(f"Non-Seizure EEG Signal {i+1}")
    plt.xlabel("Time (arbitrary units)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.ylim([-250, 250])  # Set y-axis limits to be the same for all plots

plt.tight_layout()
plt.show()

# Dimensionality reduction (PCA)
pca = PCA(n_components=45)
data_pca = pca.fit_transform(data)

# Normalize the data
data_pca = (data_pca - data_pca.mean(axis=0)) / data_pca.std(axis=0)
data_pca = np.nan_to_num(data_pca)  # Handle potential NaN or infinity values

# Load your trained model
model = load_model('epileptic_seizure_detection_model.h5')

# Predict on the entire dataset
y_pred_probs = model.predict(data_pca)
y_pred = (y_pred_probs > 0.5).astype(int).flatten()

# Confusion Matrix
conf_matrix = confusion_matrix(labels, y_pred)
print("\nConfusion Matrix:")
print(conf_matrix)

# Classification Report
print("\nClassification Report:")
print(classification_report(labels, y_pred))

# Plot the Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=["No Seizure", "Seizure"], yticklabels=["No Seizure", "Seizure"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

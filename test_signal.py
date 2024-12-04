import numpy as np
import pandas as pd
import random
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from keras.models import load_model

# Parameters
n_features = 256  # Number of original features (equivalent to number of time steps per signal, for BioAmps EEG sensor)
n_samples = 500  # Total samples to generate
seizure_proportion = 0.5  # Proportion of seizure samples
n_channels = 8  # Number of EEG channels (BioAmps often has 8 or more channels)

# Frequency bands (Delta, Theta, Alpha, Beta, Gamma)
frequency_bands = {
    'Delta': (0.5, 4),   # Slow waves (sleep)
    'Theta': (4, 8),     # Light sleep and relaxation
    'Alpha': (8, 12),    # Calm alertness
    'Beta': (12, 30),    # Active thinking, alertness
    'Gamma': (30, 40)    # High-level processing, cognition
}

# Generate synthetic EEG signals
data = []
labels = []
n_seizure = int(seizure_proportion * n_samples)
n_non_seizure = n_samples - n_seizure

# Generate Seizure and Non-Seizure signals with multiple channels
def generate_eeg_signal(seizure=False):
    # Simulate multi-channel EEG signal (8 channels)
    signal = np.zeros((n_channels, n_features))
    
    for i in range(n_channels):
        # Add random oscillations from different frequency bands
        for band, (low, high) in frequency_bands.items():
            freq = random.uniform(low, high)
            t = np.linspace(0, 1, n_features)
            signal[i] += np.sin(2 * np.pi * freq * t)  # Basic sine wave for the band
            
        if seizure:
            # Add seizure-like high-frequency activity (e.g., high Beta or Gamma oscillations)
            seizure_freq = random.uniform(20, 40)
            signal[i] += np.sin(2 * np.pi * seizure_freq * t) * random.uniform(2, 5)  # Higher amplitude for seizure signals
            
        signal[i] += np.random.normal(0, 2, n_features)  # Less noise to simulate BioAmps' higher quality signal
    
    return signal.flatten()

# Signal generation for Seizure and Non-Seizure categories
for _ in range(n_seizure):
    signal = generate_eeg_signal(seizure=True)
    data.append(signal)
    labels.append(1)  # Seizure

for _ in range(n_non_seizure):
    signal = generate_eeg_signal(seizure=False)
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

# Visualize the first 5 seizure and non-seizure signals for each channel separately
plt.figure(figsize=(15, 10))

# Plot first 5 Seizure signals
for i in range(5):
    plt.subplot(5, 2, 2*i + 1)
    plt.plot(data[labels == 1][i], label=f"Seizure Signal {i+1}")
    plt.title(f"Seizure EEG Signal {i+1}")
    plt.xlabel("Time (arbitrary units)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.ylim([-50, 50])  # Set y-axis limits to be the same for all plots

# Plot first 5 Non-Seizure signals
for i in range(5):
    plt.subplot(5, 2, 2*i + 2)
    plt.plot(data[labels == 0][i], label=f"Non-Seizure Signal {i+1}")
    plt.title(f"Non-Seizure EEG Signal {i+1}")
    plt.xlabel("Time (arbitrary units)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.ylim([-50, 50])  # Set y-axis limits to be the same for all plots

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

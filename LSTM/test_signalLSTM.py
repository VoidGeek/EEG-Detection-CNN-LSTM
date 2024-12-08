import numpy as np
import random
import seaborn as sns
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report

# Parameters
n_features = 256  # Number of original features (time steps per signal, e.g., 1 second @ 256 Hz sampling rate)
n_samples = 500  # Total samples to generate
seizure_proportion = 0.5  # Proportion of seizure samples
n_channels = 8  # Number of EEG channels (e.g., BioAmps EEG sensor with 8 channels)

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

# Function to generate EEG signal
def generate_eeg_signal(seizure=False):
    signal = np.zeros((n_channels, n_features))
    t = np.linspace(0, 1, n_features)  # 1 second duration
    
    for i in range(n_channels):
        # Add random oscillations from different frequency bands for non-seizure signals
        for band, (low, high) in frequency_bands.items():
            freq = random.uniform(low, high)
            signal[i] += np.sin(2 * np.pi * freq * t)  # Add basic sine wave

        if seizure:
            # Add seizure-like high-frequency activity and spike waves
            seizure_freq = 3  # 3 Hz spike-wave pattern (commonly seen in seizures)
            spike_wave_pattern = np.sin(2 * np.pi * seizure_freq * t) * 2  # Sharp waves with higher amplitude
            signal[i] += spike_wave_pattern
            
            # Add additional high-frequency (Gamma) activity for seizure-like chaotic behavior
            high_freq = random.uniform(30, 40)  # High frequency gamma activity
            signal[i] += np.sin(2 * np.pi * high_freq * t) * random.uniform(2, 5)  # Higher amplitude

        # Add noise to simulate EEG signal (controlled noise for non-seizure)
        if not seizure:
            signal[i] += np.random.normal(0, 1, n_features)  # Smaller noise for non-seizure
        else:
            signal[i] += np.random.normal(0, 2, n_features)  # More noise for seizure signals
    
    return signal.flatten()  # Flatten multi-channel data


# Generate seizure and non-seizure signals
for _ in range(n_seizure):
    data.append(generate_eeg_signal(seizure=True))
    labels.append(1)  # Seizure label

for _ in range(n_non_seizure):
    data.append(generate_eeg_signal(seizure=False))
    labels.append(0)  # Non-seizure label

# Convert to NumPy arrays
data = np.array(data)
labels = np.array(labels)

# Shuffle the data and labels
indices = np.arange(n_samples)
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]

# Visualization: Display 5 random seizure and non-seizure signals
plt.figure(figsize=(15, 10))

# Parameters for visualization
segment_length = 256  # Plot only the first second
amplitude_scaling = 10  # Scale to microvolt range

# Get random indices for seizure and non-seizure signals
seizure_indices = np.random.choice(np.where(labels == 1)[0], size=5, replace=False)
non_seizure_indices = np.random.choice(np.where(labels == 0)[0], size=5, replace=False)

# Plot random seizure signals
for i, idx in enumerate(seizure_indices):
    plt.subplot(5, 2, 2 * i + 1)
    signal_segment = data[idx][:segment_length] * amplitude_scaling
    plt.plot(signal_segment, label=f"Seizure Signal {i + 1}", color="red")
    plt.title(f"Random Seizure EEG Signal {i + 1}")
    plt.xlabel("Time (ms)")
    plt.ylabel("Amplitude (µV)")
    plt.legend()
    plt.ylim([-100, 100])

# Plot random non-seizure signals
for i, idx in enumerate(non_seizure_indices):
    plt.subplot(5, 2, 2 * i + 2)
    signal_segment = data[idx][:segment_length] * amplitude_scaling
    plt.plot(signal_segment, label=f"Non-Seizure Signal {i + 1}", color="blue")
    plt.title(f"Random Non-Seizure EEG Signal {i + 1}")
    plt.xlabel("Time (ms)")
    plt.ylabel("Amplitude (µV)")
    plt.legend()
    plt.ylim([-100, 100])

plt.tight_layout()
plt.show()

# Dimensionality reduction (PCA)
pca = PCA(n_components=45)
data_pca = pca.fit_transform(data)

# Reshape for LSTM model (3D input: samples, timesteps, features)
data_pca = data_pca.reshape(data_pca.shape[0], data_pca.shape[1], 1)

# Normalize the data
data_pca = (data_pca - data_pca.mean(axis=0)) / data_pca.std(axis=0)
data_pca = np.nan_to_num(data_pca)  # Handle NaN or infinity values

# Load the trained LSTM model
model = load_model('epileptic_seizure_detection_lstm_model.h5')

# Predict on the dataset
y_pred_probs = model.predict(data_pca)
y_pred = (y_pred_probs > 0.5).astype(int).flatten()

# Confusion matrix
conf_matrix = confusion_matrix(labels, y_pred)
print("\nConfusion Matrix:")
print(conf_matrix)

# Classification report
print("\nClassification Report:")
print(classification_report(labels, y_pred))

# Plot confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=["No Seizure", "Seizure"], yticklabels=["No Seizure", "Seizure"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

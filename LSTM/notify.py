import numpy as np
import matplotlib.pyplot as plt
import time
from keras.models import load_model
import random
import tkinter as tk
from tkinter import messagebox

# Parameters
n_features = 256  # Number of features (1 second of data @ 256 Hz)
n_channels = 8  # EEG channels
pca_components = 45  # Expected input dimension for the model

# Load the trained LSTM model
model = load_model('epileptic_seizure_detection_lstm_model.h5')

# Frequency bands (Delta, Theta, Alpha, Beta, Gamma)
frequency_bands = {
    'Delta': (0.5, 4),   # Slow waves (sleep)
    'Theta': (4, 8),     # Light sleep and relaxation
    'Alpha': (8, 12),    # Calm alertness
    'Beta': (12, 30),    # Active thinking, alertness
    'Gamma': (30, 40)    # High-level processing, cognition
}

# Function to generate a seizure-like EEG signal
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

# Generate a seizure signal
seizure_signal = generate_eeg_signal(seizure=True)

# Visualize the seizure signal (first second)
plt.figure(figsize=(10, 6))
plt.plot(seizure_signal[:n_features] * 10, color="red")
plt.title("Generated Seizure EEG Signal")
plt.xlabel("Time (ms)")
plt.ylabel("Amplitude (µV)")
plt.ylim([-100, 100])
plt.grid()
plt.show()

# Reshape the signal for the model (3D input: samples, timesteps, features)
seizure_signal = seizure_signal[:pca_components]  # Reduce to match expected components
seizure_signal = seizure_signal.reshape(1, pca_components, 1)

# Normalize the signal
seizure_signal = (seizure_signal - seizure_signal.mean()) / seizure_signal.std()
seizure_signal = np.nan_to_num(seizure_signal)  # Handle NaN or infinity values

# Predict using the LSTM model
pred_prob = model.predict(seizure_signal)[0, 0]
prediction = int(pred_prob > 0.5)  # 1 for seizure, 0 for no seizure

# Display notification if a seizure is detected
if prediction == 1:
    current_time = time.strftime('%H:%M:%S', time.localtime())
    print("\n⚠⚠ DANGER: Seizure Detected! ⚠⚠")
    print(f"Time: {current_time}")
    print("Immediate medical attention required!")

    # Show a pop-up notification
    root = tk.Tk()
    root.withdraw()  # Hide the main Tkinter window
    messagebox.showwarning("Seizure Alert", f"⚠⚠ DANGER: Seizure Detected at {current_time}! Immediate medical attention required!")
    root.destroy()
else:
    print("\nNo seizure detected. Patient is stable.")
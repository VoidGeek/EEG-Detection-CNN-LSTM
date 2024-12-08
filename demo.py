import numpy as np
import matplotlib.pyplot as plt

# Function to generate non-seizure (normal) EEG waves
def generate_normal_wave(duration, sampling_rate):
    t = np.linspace(0, duration, int(sampling_rate * duration))
    freq = 10  # Hz (normal brain activity frequency)
    amplitude = 20  # Normal amplitude
    noise = np.random.normal(0, 2, len(t))  # Add a bit of noise
    wave = amplitude * np.sin(2 * np.pi * freq * t) + noise
    return t, wave

# Function to generate seizure EEG waves
def generate_seizure_wave(duration, sampling_rate):
    t = np.linspace(0, duration, int(sampling_rate * duration))
    freq = np.random.choice([15, 20, 25])  # Irregular high frequencies
    amplitude = 50  # Higher amplitude
    noise = np.random.normal(0, 5, len(t))  # Add more noise
    wave = amplitude * np.sin(2 * np.pi * freq * t) + noise
    return t, wave

# Parameters
duration = 2  # seconds
sampling_rate = 250  # Hz (sampling rate typical for EEG data)

# Generate normal and seizure waves
t_normal, normal_wave = generate_normal_wave(duration, sampling_rate)
t_seizure, seizure_wave = generate_seizure_wave(duration, sampling_rate)

# Plotting the waves
plt.figure(figsize=(10, 6))

# Plot normal wave
plt.subplot(2, 1, 1)
plt.plot(t_normal, normal_wave, label="Non-Seizure Wave", color="blue")
plt.title("Non-Seizure EEG Wave")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude (µV)")
plt.legend()

# Plot seizure wave
plt.subplot(2, 1, 2)
plt.plot(t_seizure, seizure_wave, label="Seizure Wave", color="red")
plt.title("Seizure EEG Wave")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude (µV)")
plt.legend()

plt.tight_layout()
plt.show()

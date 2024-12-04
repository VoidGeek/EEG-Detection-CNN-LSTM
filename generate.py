import numpy as np
import pandas as pd
import random

# Parameters
n_features = 178  # Number of original features
n_samples = 500  # Total samples to generate
seizure_proportion = 0.5  # Proportion of seizure samples

# Generate synthetic EEG signals
data = []
n_seizure = int(seizure_proportion * n_samples)
n_non_seizure = n_samples - n_seizure

# Generate seizure signals
for _ in range(n_seizure):
    signal = np.random.uniform(-1, 1, n_features) * np.sin(np.linspace(0, 20 * np.pi, n_features)) * random.uniform(100, 300)
    signal += np.random.normal(0, 10, n_features)  # Add noise
    data.append(list(signal) + [1])  # Label: 1 for seizure

# Generate non-seizure signals
for _ in range(n_non_seizure):
    signal = np.random.uniform(-1, 1, n_features) * np.sin(np.linspace(0, 10 * np.pi, n_features)) * random.uniform(20, 80)
    signal += np.random.normal(0, 5, n_features)  # Add noise
    data.append(list(signal) + [0])  # Label: 0 for non-seizure

# Shuffle the data
random.shuffle(data)

# Create DataFrame
columns = [f"X{i+1}" for i in range(n_features)] + ["y"]
df = pd.DataFrame(data, columns=columns)

# Save to CSV
output_file = "sdata.csv"
df.to_csv(output_file, index=False)
print(f"Synthetic EEG data saved to {output_file}")

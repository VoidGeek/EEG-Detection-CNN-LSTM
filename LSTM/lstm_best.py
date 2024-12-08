import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from keras import models, layers
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv('data.csv')

# Prepare input features and binary labels
X = df.iloc[:, 1:-1].values  # Exclude ID and label columns
y = df['y'].values           # Labels

# Convert to binary classification: Seizure (1) vs. No Seizure (0)
y = np.where(y == 1, 1, 0)

# Dimensionality reduction (reduce features to 45)
pca = PCA(n_components=45)
X = pca.fit_transform(X)

# Split into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=1)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=1)

# Save test set for future use
np.save('X_test1.npy', X_test)
np.save('y_test1.npy', y_test)

# Normalize input data
X_train = (X_train - X_train.mean(axis=0)) / X_train.std(axis=0)
X_val = (X_val - X_val.mean(axis=0)) / X_val.std(axis=0)
X_test = (X_test - X_test.mean(axis=0)) / X_test.std(axis=0)

# Handle potential NaN or infinity values in normalized data
X_train = np.nan_to_num(X_train)
X_val = np.nan_to_num(X_val)
X_test = np.nan_to_num(X_test)

# Reshape input data for LSTM
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Define the LSTM model
model = models.Sequential([
    layers.Input(shape=(45, 1)),  # 45 features, 1 timestep
    layers.LSTM(128, return_sequences=True),
    layers.LSTM(64),
    layers.Dense(32, activation='relu'),
    layers.Dense(16, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Binary classification output
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    x=X_train,
    y=y_train,
    batch_size=10,
    epochs=10,  # Use smaller epochs for testing; adjust as needed
    verbose=2,
    validation_data=(X_val, y_val)
)

# Save the trained model
model.save('epileptic_seizure_detection_lstm_model.h5')
print("Model saved as 'epileptic_seizure_detection_lstm_model.h5'")

# Predictions and Confusion Matrix for Training Data
y_train_pred_probs = model.predict(X_train)
y_train_pred = (y_train_pred_probs > 0.5).astype(int).flatten()
conf_matrix_train = confusion_matrix(y_train, y_train_pred)
print("\nTraining Confusion Matrix:")
print(conf_matrix_train)

plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix_train, annot=True, fmt='d', cmap='Blues', xticklabels=["No Seizure", "Seizure"], yticklabels=["No Seizure", "Seizure"])
plt.title("Training Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# Predictions and Confusion Matrix for Validation Data
y_val_pred_probs = model.predict(X_val)
y_val_pred = (y_val_pred_probs > 0.5).astype(int).flatten()
conf_matrix_val = confusion_matrix(y_val, y_val_pred)
print("\nValidation Confusion Matrix:")
print(conf_matrix_val)

plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix_val, annot=True, fmt='d', cmap='Blues', xticklabels=["No Seizure", "Seizure"], yticklabels=["No Seizure", "Seizure"])
plt.title("Validation Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

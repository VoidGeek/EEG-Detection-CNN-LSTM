import os
import numpy as np
from keras import models, layers, optimizers, callbacks, utils
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

def build_cnn_lstm(input_shape):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv1D(filters=64, kernel_size=3, activation='relu'),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.3),
        layers.Conv1D(filters=128, kernel_size=3, activation='relu'),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.3),
        layers.LSTM(64, return_sequences=True),
        layers.LSTM(32),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(2, activation='softmax')
    ])
    model.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(model_save_path,features_file,labels_file):
    print("[INFO] Loading preprocessed data.")
    if not os.path.exists(features_file) or not os.path.exists(labels_file):
        raise FileNotFoundError("[ERROR] Preprocessed data not found. Please run the preprocessing script first.")
    
    X = np.load(features_file)
    y = np.load(labels_file)

    print("[INFO] Splitting data into training and testing sets.")
    X_flat = X.reshape(X.shape[0], -1)
    X_train, X_test, y_train, y_test = train_test_split(X_flat, y, test_size=0.3, random_state=42, stratify=y)

    print("[INFO] Applying SMOTE to balance training data.")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    X_train_resampled = X_train_resampled.reshape(X_train_resampled.shape[0], X.shape[1], 1)
    y_train_resampled = utils.to_categorical(y_train_resampled, 2)

    X_test = X_test.reshape(X_test.shape[0], X.shape[1], 1)
    y_test = utils.to_categorical(y_test, 2)

    print("[INFO] Building and training CNN-LSTM model.")
    model = build_cnn_lstm(X_train_resampled.shape[1:])
    early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    model.fit(X_train_resampled, y_train_resampled, validation_split=0.2, epochs=20, batch_size=64, callbacks=[early_stopping])

    print("[INFO] Saving the model.")
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    model.save(model_save_path)
    print(f"[INFO] Model saved to {model_save_path}")

    return model, X_test, y_test

if __name__ == "__main__":
    model_save_path = "../model/model.h5"
    features_file = "../processed_data/features.npy"
    labels_file = "../processed_data/labels.npy"
    print("[INFO] Starting training process.")
    model, X_test, y_test = train_model(model_save_path,features_file, labels_file)
    print("[INFO] Training completed successfully.")

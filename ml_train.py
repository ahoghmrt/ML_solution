import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd

# -------------------------------
# Load training dataset (.npz)
# -------------------------------
data = np.load("ml_training_data/training_data.npz")
X_raw = data["waveforms"]
y = data["labels"].reshape(X_raw.shape[0], -1)  # Flattened (t0, amplitude) pairs
time = data["time"]

# -------------------------------
# Baseline Subtraction (Rolling Quantile)
# -------------------------------
def rolling_quantile_baseline(waveform, window_size=31, quantile=0.1):
    if window_size % 2 == 0:
        window_size += 1
    baseline = pd.Series(waveform).rolling(window=window_size, center=True, min_periods=1).quantile(quantile)
    return baseline.values

# Apply baseline subtraction to all waveforms
X_baseline_subtracted = np.array([
    waveform - rolling_quantile_baseline(waveform, window_size=31, quantile=0.1)
    for waveform in X_raw
])

# Normalize input waveforms
X_mean = X_baseline_subtracted.mean(axis=1, keepdims=True)
X_std = X_baseline_subtracted.std(axis=1, keepdims=True)
X_norm = (X_baseline_subtracted - X_mean) / (X_std + 1e-8)

# -------------------------------
# Split Train/Validation
# -------------------------------
X_train, X_val, y_train, y_val = train_test_split(X_norm, y, test_size=0.2, random_state=42)

# -------------------------------
# Define ML Model
# -------------------------------
model = keras.Sequential([
    layers.Input(shape=(X_train.shape[1],)),
    layers.Dense(128, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(y_train.shape[1])  # Output (t0_1, A1, t0_2, A2, ...)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# -------------------------------
# Train Model
# -------------------------------
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=32,
    verbose=1
)

# -------------------------------
# Evaluate and Plot Results
# -------------------------------
y_val_pred = model.predict(X_val)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(y_val[:, 0], y_val_pred[:, 0], alpha=0.6)
plt.xlabel("True t0 (signal 1)")
plt.ylabel("Predicted t0")
plt.title("t0 Prediction")

plt.subplot(1, 2, 2)
plt.scatter(y_val[:, 1], y_val_pred[:, 1], alpha=0.6)
plt.xlabel("True Amplitude (signal 1)")
plt.ylabel("Predicted Amplitude")
plt.title("Amplitude Prediction")

plt.tight_layout()
plt.savefig("ml_training_data/training_evaluation_plot.png")
plt.close()

# -------------------------------
# Save Model (Keras format)
# -------------------------------
model.save("ml_training_data/signal_extraction_model.keras")
print("\n✅ Model retrained using baseline-subtracted waveforms.")
print("📊 Training evaluation plot saved: ml_training_data/training_evaluation_plot.png")
print("💾 Model saved as: signal_extraction_model.keras")

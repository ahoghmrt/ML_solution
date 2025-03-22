import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os

# -------------------------------
# Load ML Dataset
# -------------------------------
data = np.load("ml_training_data/training_data.npz")
X = data["waveforms"]
y = data["labels"].reshape(X.shape[0], -1)
time = data["time"]

# -------------------------------
# Normalize Inputs
# -------------------------------
X_mean = X.mean(axis=1, keepdims=True)
X_std = X.std(axis=1, keepdims=True)
X_norm = (X - X_mean) / (X_std + 1e-8)

# -------------------------------
# Train/Validation Split
# -------------------------------
X_train, X_val, y_train, y_val = train_test_split(X_norm, y, test_size=0.2, random_state=42)

# -------------------------------
# Define Model Architecture
# -------------------------------
model = keras.Sequential([
    layers.Reshape((X.shape[1], 1), input_shape=(X.shape[1],)),
    layers.Conv1D(32, 3, activation='relu'),
    layers.MaxPooling1D(2),
    layers.Conv1D(64, 3, activation='relu'),
    layers.GlobalMaxPooling1D(),
    layers.Dense(64, activation='relu'),
    layers.Dense(y.shape[1])
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
# Evaluate & Save Results
# -------------------------------
y_pred = model.predict(X_val)
os.makedirs("ml_training_data", exist_ok=True)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(y_val[:, 0], y_pred[:, 0], alpha=0.5)
plt.xlabel("True t0")
plt.ylabel("Predicted t0")
plt.title("t0 Prediction")

plt.subplot(1, 2, 2)
plt.scatter(y_val[:, 1], y_pred[:, 1], alpha=0.5)
plt.xlabel("True Amplitude")
plt.ylabel("Predicted Amplitude")
plt.title("Amplitude Prediction")

plt.tight_layout()
plt.savefig("ml_training_data/training_evaluation_plot.png")
plt.close()

# Save model
model.save("ml_training_data/signal_extraction_model.keras")
print("\n✅ Model training complete and saved as 'signal_extraction_model.keras'")

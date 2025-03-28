import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# ---------------------------
# Load Dataset
# ---------------------------
data = np.load("ml_training_data/training_data_counts.npz")
X = data["waveforms"]  # shape (n_samples, n_time_bins)
y = data["labels"]     # shape (n_samples,) - integers [0, 1, ..., max_signals]

print(f"✅ Loaded dataset: {X.shape[0]} samples, {X.shape[1]} time bins per waveform")
print(f"✅ Target shape: {y.shape}, Min: {y.min()}, Max: {y.max()}")

# ---------------------------
# Split Train / Validation
# ---------------------------
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------------------
# Build Model
# ---------------------------
model = keras.Sequential([
    layers.Input(shape=(X.shape[1],)),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(np.max(y) + 1, activation='softmax')  # number of classes = max signals + 1
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ---------------------------
# Train Model
# ---------------------------
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=45,
    batch_size=64,
    verbose=1
)

# ---------------------------
# Save Model
# ---------------------------
model.save("signal_count_model.keras")
print("✅ Model saved as signal_count_model.keras")

# ---------------------------
# Plot Training History
# ---------------------------
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Curve")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["accuracy"], label="Train Acc")
plt.plot(history.history["val_accuracy"], label="Val Acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy Curve")
plt.legend()

plt.tight_layout()
plt.savefig("training_plots/count_model_training.png")
plt.show()

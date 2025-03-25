import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# -----------------------------
# Load Dataset
# -----------------------------
data = np.load("ml_training_data/training_data_signals.npz")
X = data["waveforms"]
y = data["labels"]  # shape: (samples, max_signals, 2)
time = data["time"]

max_signals = y.shape[1]

print(f"✅ Loaded dataset: {X.shape[0]} samples, each with {X.shape[1]} time bins")
print(f"✅ Label shape (t0, A): {y.shape}, Time shape: {time.shape}")

# -----------------------------
# Load signal_count_model to use predicted signal counts
# -----------------------------
count_model = keras.models.load_model("signal_count_model.keras")
count_preds = np.argmax(count_model.predict(X), axis=1)

# -----------------------------
# Prepare masked labels using predicted signal counts
# -----------------------------
y_masked = np.zeros_like(y)

for i in range(len(X)):
    count = count_preds[i]
    y_masked[i, :count, :] = y[i, :count, :]

# -----------------------------
# Build Model
# -----------------------------
model = keras.Sequential([
    layers.Input(shape=(X.shape[1],)),
    layers.Dense(128, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(max_signals * 2),
    layers.Reshape((max_signals, 2), name="signal_output")
])

model.compile(optimizer='adam',
              loss='mae',
              metrics=['mae'])

model.summary()

# -----------------------------
# Train
# -----------------------------
history = model.fit(
    X, y_masked,
    validation_split=0.2,
    epochs=30,
    batch_size=64
)

# -----------------------------
# Save Model
# -----------------------------
model.save("signal_model_using_counts.keras")
print("✅ Saved: signal_model_using_counts.keras")

# -----------------------------
# Plot training history
# -----------------------------
os.makedirs("training_plots", exist_ok=True)
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title("Signal Model Training (using predicted signal counts)")
plt.xlabel("Epoch")
plt.ylabel("MAE Loss")
plt.legend()
plt.grid(True)
plt.savefig("training_plots/signal_model_using_counts_training.png")
plt.show()

import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import pandas as pd

# -----------------------------
# Load Dataset
# -----------------------------
data = np.load("ml_training_data/training_data_signals.npz")
X = data["waveforms"]
y = data["labels"]  # shape: (samples, max_signals, 2)
time = data["time"]
max_signals = y.shape[1]

print(f"✅ Loaded dataset: {X.shape[0]} samples, each with {X.shape[1]} time bins")

# Normalize waveforms (mean 0, std 1)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# -----------------------------
# Load signal_count_model
# -----------------------------
count_model = keras.models.load_model("signal_count_model.keras")
count_preds = np.argmax(count_model.predict(X), axis=1)

# -----------------------------
# Create masked labels and mask
# -----------------------------
y_masked = np.zeros_like(y)
mask = np.zeros((len(X), max_signals, 1))

for i in range(len(X)):
    count = count_preds[i]
    y_masked[i, :count, :] = y[i, :count, :]
    mask[i, :count, 0] = 1

# Combine labels and mask → shape: (samples, max_signals, 3)
y_with_mask = np.concatenate([y_masked, mask], axis=-1)

# -----------------------------
# Custom masked MAE loss
# -----------------------------
class MaskedMAELoss(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        y_true_vals = y_true[:, :, :2]
        mask = y_true[:, :, 2:]
        error = tf.abs(y_true_vals - y_pred)
        masked_error = error * mask
        return tf.reduce_sum(masked_error) / tf.reduce_sum(mask)

# -----------------------------
# Custom metrics
# -----------------------------
def masked_mae_t0(y_true, y_pred):
    y_true_t0 = y_true[:, :, 0:1]
    y_pred_t0 = y_pred[:, :, 0:1]
    mask = y_true[:, :, 2:3]
    error = tf.abs(y_true_t0 - y_pred_t0)
    masked_error = error * mask
    return tf.reduce_sum(masked_error) / tf.reduce_sum(mask)

def masked_mae_amp(y_true, y_pred):
    y_true_amp = y_true[:, :, 1:2]
    y_pred_amp = y_pred[:, :, 1:2]
    mask = y_true[:, :, 2:3]
    error = tf.abs(y_true_amp - y_pred_amp)
    masked_error = error * mask
    return tf.reduce_sum(masked_error) / tf.reduce_sum(mask)

# -----------------------------
# Build Model
# -----------------------------
input_layer = layers.Input(shape=(X.shape[1], 1))

x = layers.Conv1D(32, kernel_size=5, activation='relu', padding='same')(input_layer)
x = layers.Conv1D(64, kernel_size=5, activation='relu', padding='same')(x)
x = layers.Flatten()(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dense(max_signals * 2)(x)
output = layers.Reshape((max_signals, 2), name="signal_output")(x)

model = keras.Model(inputs=input_layer, outputs=output)
model.compile(
    optimizer='adam',
    loss=MaskedMAELoss(),
    metrics=[masked_mae_t0, masked_mae_amp]
)
model.summary()

# -----------------------------
# Train
# -----------------------------
history = model.fit(
    X[..., np.newaxis],
    y_with_mask,
    validation_split=0.2,
    epochs=30,
    batch_size=64
)

# -----------------------------
# Save Model
# -----------------------------
model.save("signal_model_conv_masked.keras")
print("✅ Saved: signal_model_conv_masked.keras")

# -----------------------------
# Save training history to CSV
# -----------------------------
history_df = pd.DataFrame(history.history)
csv_path = "training_plots/signal_model_conv_masked_history.csv"
history_df.to_csv(csv_path, index=False)
print(f"📁 Saved training history to: {csv_path}")

# -----------------------------
# Plot training history
# -----------------------------
os.makedirs("training_plots", exist_ok=True)

plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.plot(history.history['masked_mae_t0'], label='Train t₀ MAE')
plt.plot(history.history['val_masked_mae_t0'], label='Val t₀ MAE')
plt.plot(history.history['masked_mae_amp'], label='Train Amp MAE')
plt.plot(history.history['val_masked_mae_amp'], label='Val Amp MAE')
plt.title("Signal Model Training (Masked MAE, Conv1D)")
plt.xlabel("Epoch")
plt.ylabel("Loss / MAE")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("training_plots/signal_model_conv_masked_training.png")
plt.show()

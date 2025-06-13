import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split

# -----------------------------
# Load Dataset
# -----------------------------
data = np.load("ml_training_data/training_data_signals.npz")
X_wave = data["waveforms"]                     # shape: (samples, 120)
y = data["labels"]                             # shape: (samples, max_signals, 2)
time = data["time"]

max_signals = y.shape[1]

print(f"✅ Loaded dataset: {X_wave.shape[0]} samples, each with {X_wave.shape[1]} time bins")
print(f"✅ Label shape (t0, A): {y.shape}, Time shape: {time.shape}")

# -----------------------------
# Normalize targets (t0 and amplitude separately)
# -----------------------------
t0s = y[:, :, 0]
amps = y[:, :, 1]

scaler_t0 = StandardScaler()
scaler_amp = StandardScaler()

t0s_norm = scaler_t0.fit_transform(t0s)
amps_norm = scaler_amp.fit_transform(amps)

y_norm = np.stack([t0s_norm, amps_norm], axis=-1)
y_flat = y_norm.reshape((y.shape[0], max_signals * 2))

# Save target scalers
os.makedirs("training_plots", exist_ok=True)
joblib.dump(scaler_t0, "training_plots/t0_scaler.pkl")
joblib.dump(scaler_amp, "training_plots/amp_scaler.pkl")

# -----------------------------
# Normalize inputs (waveforms)
# -----------------------------
scaler_wave = StandardScaler()
X_wave_scaled = scaler_wave.fit_transform(X_wave)
X_wave_scaled = np.expand_dims(X_wave_scaled, axis=-1)

# Save waveform scaler
joblib.dump(scaler_wave, "training_plots/waveform_scaler.pkl")


input_wave = layers.Input(shape=(X_wave.shape[1], 1), name="waveform_input")

# Conv1D processing
x = layers.Conv1D(32, kernel_size=5, activation='relu', padding='same')(input_wave)
x = layers.Conv1D(64, kernel_size=5, activation='relu', padding='same')(x)
x = layers.Flatten()(x)

# Dense processing
x = layers.Dense(128, activation='relu')(x)
x = layers.Dense(64, activation='relu')(x)
output = layers.Dense(max_signals * 2, name="signal_output")(x)

model = keras.Model(inputs=input_wave, outputs=output)
model.compile(optimizer='adam', loss='mae', metrics=['mae'])
model.summary()


# -----------------------------
# Train/Test Split
# -----------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X_wave_scaled, y_flat, test_size=0.35, random_state=42
)

# -----------------------------
# Train
# -----------------------------
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=64
)

# -----------------------------
# Save Model and History
# -----------------------------
model.save("signal_model.keras")
pd.DataFrame(history.history).to_csv("training_plots/signal_model_history.csv", index=False)

# -----------------------------
# Plot Training History
# -----------------------------
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Train Loss (MAE)')
plt.plot(history.history['val_loss'], label='Val Loss (MAE)')
plt.title("Signal Model Training")
plt.xlabel("Epoch")
plt.ylabel("Loss (MAE)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("training_plots/signal_model_training.png")
plt.show()

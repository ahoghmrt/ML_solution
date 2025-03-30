import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler
import pandas as pd

# -----------------------------
# Load Dataset
# -----------------------------
data = np.load("ml_training_data/training_data_counts.npz")
X_wave = data["waveforms"]                 # shape: (samples, 120)
X_t0 = data["candidate_t0s"]               # shape: (samples, 5)
X_amp = data["candidate_amps"]             # shape: (samples, 5)
y = data["labels"]                         # signal counts: (samples,)
time = data["time"]

# Normalize waveforms
scaler_wave = StandardScaler()
X_wave = scaler_wave.fit_transform(X_wave)

# Normalize candidate features (optional but helpful)
scaler_t0 = StandardScaler()
X_t0 = scaler_t0.fit_transform(X_t0)

scaler_amp = StandardScaler()
X_amp = scaler_amp.fit_transform(X_amp)

# -----------------------------
# Build Model
# -----------------------------
input_wave = layers.Input(shape=(X_wave.shape[1],), name="waveform_input")
input_t0 = layers.Input(shape=(X_t0.shape[1],), name="t0_input")
input_amp = layers.Input(shape=(X_amp.shape[1],), name="amp_input")

# Waveform branch
x_wave = layers.Dense(128, activation='relu')(input_wave)
x_wave = layers.Dense(64, activation='relu')(x_wave)

# Candidate feature branch
x_feat = layers.Concatenate()([input_t0, input_amp])
x_feat = layers.Dense(64, activation='relu')(x_feat)

# Combine both branches
x = layers.Concatenate()([x_wave, x_feat])
x = layers.Dense(64, activation='relu')(x)
output = layers.Dense(7, activation='softmax', name="count_output")(x)

model = keras.Model(inputs=[input_wave, input_t0, input_amp], outputs=output)

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# -----------------------------
# Train
# -----------------------------
history = model.fit(
    [X_wave, X_t0, X_amp],
    y,
    validation_split=0.2,
    epochs=30,
    batch_size=64
)

# -----------------------------
# Save Model and History
# -----------------------------
model.save("signal_count_model_with_candidates.keras")
os.makedirs("training_plots", exist_ok=True)
pd.DataFrame(history.history).to_csv("training_plots/count_model_with_candidates_history.csv", index=False)

# -----------------------------
# Plot Training History
# -----------------------------
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title("Signal Count Model Training")
plt.xlabel("Epoch")
plt.ylabel("Loss / Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("training_plots/count_model_with_candidates.png")
plt.show()

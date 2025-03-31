import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd

# -----------------------------
# Load Dataset
# -----------------------------
data = np.load("ml_training_data/training_data_counts.npz")
X = data["waveforms"]                      # shape: (samples, 120)
y = data["labels"]                         # shape: (samples,)
time = data["time"]

print(f"✅ Loaded: {X.shape[0]} waveforms with {X.shape[1]} time bins")
print(f"✅ Label shape (signal counts): {y.shape}")

# -----------------------------
# Normalize Inputs
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# Build Model (Input = waveform, Output = signal count class)
# -----------------------------
input_layer = layers.Input(shape=(X.shape[1],), name="waveform_input")
x = layers.Dense(128, activation='relu')(input_layer)
x = layers.Dense(64, activation='relu')(x)
x = layers.Dense(32, activation='relu')(x)
output = layers.Dense(4, activation='softmax', name="count_output")(x)  # predict 0–6

model = keras.Model(inputs=input_layer, outputs=output)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# -----------------------------
# Train/Test Split
# -----------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
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
os.makedirs("training_plots", exist_ok=True)
model.save("signal_count_model.keras")
pd.DataFrame(history.history).to_csv("training_plots/signal_count_model_history.csv", index=False)

# -----------------------------
# Plot Training History
# -----------------------------
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title("Signal Count Model Training")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("training_plots/signal_count_model_training.png")
plt.show()

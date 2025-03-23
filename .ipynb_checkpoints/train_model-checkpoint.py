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

dataset_path = "ml_training_data/training_data.npz"
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

data = np.load(dataset_path)
X = data["waveforms"]
y = data["labels"].reshape(X.shape[0], -1)
time = data["time"]

print(f"✅ Loaded dataset: {X.shape[0]} samples, each with {X.shape[1]} time bins")
print(f"✅ Label shape: {y.shape}, Time shape: {time.shape}")

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
    layers.Input(shape=(X.shape[1],)),
    layers.Dense(128, activation='relu'),
    layers.Dense(128, activation='relu'),
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
# Evaluate & Save Results using baseline-subtracted waveforms and truth labels
# Calculate accuracy in numbers
# -------------------------------
# Predict using baseline-subtracted validation waveforms
# Compare predictions to ground truth time-amplitude pairs from truth files
y_pred = model.predict(X_val)
os.makedirs("ml_training_data", exist_ok=True)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
for i in range(0, y.shape[1], 2):
    plt.scatter(y_val[:, i], y_pred[:, i], alpha=0.5, label=f'Signal {i//2 + 1}')
plt.xlabel("True t0")
plt.ylabel("Predicted t0")
plt.title("t0 Prediction")
plt.legend()

plt.subplot(1, 2, 2)
for i in range(1, y.shape[1], 2):
    plt.scatter(y_val[:, i], y_pred[:, i], alpha=0.5, label=f'Signal {i//2 + 1}')
plt.xlabel("True Amplitude")
plt.ylabel("Predicted Amplitude")
plt.title("Amplitude Prediction")
plt.legend()

plt.tight_layout()
plt.savefig("ml_training_data/training_evaluation_plot.png")
plt.close()

from sklearn.metrics import mean_absolute_error, mean_squared_error

for i in range(y.shape[1]):
    mae = mean_absolute_error(y_val[:, i], y_pred[:, i])
    rmse = np.sqrt(mean_squared_error(y_val[:, i], y_pred[:, i]))
    param = "t0" if i % 2 == 0 else "Amplitude"
    signal_num = (i // 2) + 1
    print(f"📊 Signal {signal_num} - {param}: MAE = {mae:.3f}, RMSE = {rmse:.3f}")

# Save model
model.save("ml_training_data/signal_extraction_model.keras")
print("\n✅ Model training complete and saved as 'signal_extraction_model.keras'")

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.utils import to_categorical

# -------------------------------
# Load ML Dataset
# -------------------------------
dataset_path = "ml_training_data/training_data.npz"
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

data = np.load(dataset_path)
X = data["waveforms"]
y = data["labels"].reshape(X.shape[0], -1)  # (n_samples, max_signals*3)
time = data["time"]
count = data["counts"]

print(f"✅ Loaded dataset: {X.shape[0]} samples, each with {X.shape[1]} time bins")
print(f"✅ Label shape: {y.shape}, Count shape: {count.shape}, Time shape: {time.shape}")

# -------------------------------
# Normalize Inputs
# -------------------------------
X_mean = X.mean(axis=1, keepdims=True)
X_std = X.std(axis=1, keepdims=True)
X_norm = (X - X_mean) / (X_std + 1e-8)

# -------------------------------
# Encode Signal Count as Classification
# -------------------------------
max_signals = y.shape[1] // 3
num_classes = max_signals + 1
count_clipped = np.clip(count, 0, max_signals).astype(int)
count_categorical = to_categorical(count_clipped, num_classes=num_classes)

# -------------------------------
# Train/Validation Split
# -------------------------------
X_train, X_val, y_train, y_val, count_train, count_val = train_test_split(
    X_norm, y, count_categorical, test_size=0.2, random_state=42
)

# -------------------------------
# Define Model Architecture (t0, A, Presence + Signal Count Classification)
# -------------------------------
input_layer = layers.Input(shape=(X.shape[1],))
x = layers.Reshape((X.shape[1], 1))(input_layer)
x = layers.Conv1D(32, kernel_size=3, activation='relu')(x)
x = layers.MaxPooling1D(pool_size=2)(x)
x = layers.Conv1D(64, kernel_size=3, activation='relu')(x)
x = layers.GlobalMaxPooling1D()(x)
x = layers.Dense(64, activation='relu')(x)

signal_output = layers.Dense(y.shape[1], activation='linear', name='signal_output')(x)
count_output = layers.Dense(num_classes, activation='softmax', name='count_output')(x)

model = keras.Model(inputs=input_layer, outputs=[signal_output, count_output])

model.compile(
    optimizer='adam',
    loss={'signal_output': 'mse', 'count_output': 'categorical_crossentropy'},
    metrics={'signal_output': 'mae', 'count_output': 'accuracy'},
    loss_weights={'signal_output': 1.0, 'count_output': 3.0}
)

# -------------------------------
# Train Model
# -------------------------------
history = model.fit(
    X_train,
    {'signal_output': y_train, 'count_output': count_train},
    validation_data=(X_val, {'signal_output': y_val, 'count_output': count_val}),
    epochs=30,
    batch_size=32,
    verbose=1
)

# -------------------------------
# Evaluate & Save Results
# -------------------------------
pred_signal, pred_count = model.predict(X_val)
pred_signal = pred_signal.reshape((-1, max_signals, 3))  # reshape to (samples, signals, 3)
pred_count_labels = np.argmax(pred_count, axis=1)
true_count_labels = np.argmax(count_val, axis=1)

os.makedirs("ml_training_data", exist_ok=True)

# -------------------------------
# Accuracy Metrics
# -------------------------------
for i in range(max_signals):
    mae_t0 = mean_absolute_error(y_val[:, i*3], pred_signal[:, i, 0])
    rmse_t0 = np.sqrt(mean_squared_error(y_val[:, i*3], pred_signal[:, i, 0]))
    mae_amp = mean_absolute_error(y_val[:, i*3+1], pred_signal[:, i, 1])
    rmse_amp = np.sqrt(mean_squared_error(y_val[:, i*3+1], pred_signal[:, i, 1]))
    print(f"📊 Signal {i+1} - t0: MAE = {mae_t0:.3f}, RMSE = {rmse_t0:.3f}")
    print(f"📊 Signal {i+1} - Amplitude: MAE = {mae_amp:.3f}, RMSE = {rmse_amp:.3f}")

# -------------------------------
# Signal Count Prediction Evaluation
# -------------------------------
print("\n📊 Predicted vs True Signal Counts (sample preview):")
for i in range(10):
    print(f"Sample {i}: Predicted = {pred_count_labels[i]}, True = {true_count_labels[i]}")

# Save model
model.save("ml_training_data/signal_extraction_model.keras")
print("\n✅ Model training complete and saved as 'signal_extraction_model.keras'")

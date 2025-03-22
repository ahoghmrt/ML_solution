import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os

# -------------------------------
# Load Trained Model
# -------------------------------
model = tf.keras.models.load_model("ml_training_data/signal_extraction_model.keras")

# -------------------------------
# Load Data to Predict
# -------------------------------
data = np.load("ml_training_data/training_data.npz")
X = data["waveforms"]
y_true = data["labels"]
time = data["time"]

# -------------------------------
# Normalize Input
# -------------------------------
X_mean = X.mean(axis=1, keepdims=True)
X_std = X.std(axis=1, keepdims=True)
X_norm = (X - X_mean) / (X_std + 1e-8)

# -------------------------------
# Predict Using Model
# -------------------------------
y_pred = model.predict(X_norm)

# -------------------------------
# Plot Example Prediction
# -------------------------------
sample_id = 0
waveform = X[sample_id]
pred_signals = y_pred[sample_id].reshape(-1, 2)
true_signals = y_true[sample_id]

plt.figure(figsize=(12, 5))
plt.plot(time, waveform, label="Waveform", linewidth=1.5)

for i, (t0, amp) in enumerate(true_signals):
    if amp > 0:
        plt.plot(t0, amp, 'go', label=f"True Signal {i+1}" if i == 0 else "")

for i, (t0, amp) in enumerate(pred_signals):
    if amp > 0:
        plt.plot(t0, amp, 'rx', label=f"Predicted Signal {i+1}" if i == 0 else "")

plt.xlabel("Time (ns)")
plt.ylabel("Amplitude")
plt.title(f"Predicted vs True Signals - Sample {sample_id}")
plt.grid(True)
plt.legend()

os.makedirs("predictions", exist_ok=True)
plt.savefig(f"predictions/prediction_sample_{sample_id}.png")
plt.close()
print(f"✅ Prediction plot saved: predictions/prediction_sample_{sample_id}.png")

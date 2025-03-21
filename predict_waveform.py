import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os

# -------------------------------
# Load Trained Model
# -------------------------------
model = tf.keras.models.load_model("ml_training_data/signal_extraction_model.keras")

# -------------------------------
# Load New Data
# -------------------------------
data = np.load("ml_training_data/training_data.npz")  # can be replaced with new data
X_raw = data["waveforms"]
y_true = data["labels"]
time = data["time"]

# -------------------------------
# Normalize Input Same as Training
# -------------------------------
X_mean = X_raw.mean(axis=1, keepdims=True)
X_std = X_raw.std(axis=1, keepdims=True)
X_norm = (X_raw - X_mean) / (X_std + 1e-8)

# -------------------------------
# Predict Using Model
# -------------------------------
y_pred = model.predict(X_norm)
y_pred = y_pred.reshape(y_pred.shape[0], -1, 2)  # (samples, max_signals, 2)

# -------------------------------
# Plot Example Prediction
# -------------------------------
sample_id = 0
waveform = X_raw[sample_id]
pred_signals = y_pred[sample_id]
true_signals = y_true[sample_id]

plt.figure(figsize=(12, 5))
plt.plot(time, waveform, label="Waveform", linewidth=1.5)

for t0, amp in true_signals:
    if amp > 0:
        plt.plot(t0, amp, 'go', label="True Signal" if 'True Signal' not in plt.gca().get_legend_handles_labels()[1] else "")

for t0, amp in pred_signals:
    if amp > 0:
        plt.plot(t0, amp, 'rx', label="Predicted Signal" if 'Predicted Signal' not in plt.gca().get_legend_handles_labels()[1] else "")

plt.title(f"Predicted vs True Signals - Sample {sample_id}")
plt.xlabel("Time (ns)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.legend()
os.makedirs("predictions", exist_ok=True)
plt.savefig(f"predictions/prediction_sample_{sample_id}.png")
plt.close()
print(f"✅ Prediction plot saved: predictions/prediction_sample_{sample_id}.png")


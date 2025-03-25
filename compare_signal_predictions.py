import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow import keras

# ----------------------------
# Load prediction model and data
# ----------------------------
signal_model = keras.models.load_model("signal_model_using_counts.keras")
count_model = keras.models.load_model("signal_count_model.keras")

data = np.load("ml_training_data/training_data_signals.npz")
X = data["waveforms"]
y_true = data["labels"]  # shape (n_samples, max_signals, 2)
time = data["time"]

# ----------------------------
# Predict counts and signals
# ----------------------------
pred_counts = np.argmax(count_model.predict(X), axis=1)
pred_signals = signal_model.predict(X)

# ----------------------------
# Prepare comparison arrays
# ----------------------------
pred_t0s_all, true_t0s_all = [], []
pred_amps_all, true_amps_all = [], []

for i in range(len(X)):
    count = pred_counts[i]
    for j in range(count):
        pred_t0, pred_amp = pred_signals[i][j]
        true_t0, true_amp = y_true[i][j]
        pred_t0s_all.append(pred_t0)
        pred_amps_all.append(pred_amp)
        true_t0s_all.append(true_t0)
        true_amps_all.append(true_amp)

# ----------------------------
# Plot scatter comparison
# ----------------------------
os.makedirs("comparison_plots", exist_ok=True)

plt.figure(figsize=(10, 6))
plt.scatter(true_t0s_all, pred_t0s_all, alpha=0.6, edgecolors='k')
plt.plot([min(true_t0s_all), max(true_t0s_all)], [min(true_t0s_all), max(true_t0s_all)], 'r--', label="Ideal")
plt.xlabel("True t0 (ns)")
plt.ylabel("Predicted t0 (ns)")
plt.title("Scatter Plot: True vs Predicted t0")
plt.legend()
plt.grid(True)
plt.savefig("comparison_plots/true_vs_predicted_t0.png")
plt.close()

plt.figure(figsize=(10, 6))
plt.scatter(true_amps_all, pred_amps_all, alpha=0.6, edgecolors='k')
plt.plot([min(true_amps_all), max(true_amps_all)], [min(true_amps_all), max(true_amps_all)], 'r--', label="Ideal")
plt.xlabel("True Amplitude")
plt.ylabel("Predicted Amplitude")
plt.title("Scatter Plot: True vs Predicted Amplitude")
plt.legend()
plt.grid(True)
plt.savefig("comparison_plots/true_vs_predicted_amplitude.png")
plt.close()

print("✅ Saved comparison plots in 'comparison_plots/' folder")

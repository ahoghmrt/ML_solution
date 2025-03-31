import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow import keras
import joblib

# ----------------------------
# Load prediction models
# ----------------------------
signal_model = keras.models.load_model("signal_model.keras")
count_model = keras.models.load_model("signal_count_model.keras")

# ----------------------------
# Load data
# ----------------------------
data = np.load("ml_training_data/training_data_signals.npz")
X = data["waveforms"]           # (samples, 120)
y_true = data["labels"]         # (samples, max_signals, 2)
time = data["time"]

# ----------------------------
# Load scalers for inverse transform
# ----------------------------
scaler_wave = joblib.load("training_plots/waveform_scaler.pkl")
scaler_t0 = joblib.load("training_plots/t0_scaler.pkl")
scaler_amp = joblib.load("training_plots/amp_scaler.pkl")

# ----------------------------
# Normalize inputs using saved scaler
# ----------------------------
X_scaled = scaler_wave.transform(X)

# ----------------------------
# Predict counts and signals
# ----------------------------
pred_counts = np.argmax(count_model.predict(X_scaled), axis=1)
pred_signals_norm = signal_model.predict(X_scaled[..., np.newaxis])

# ----------------------------
# Inverse transform predictions
# ----------------------------
pred_signals = pred_signals_norm.copy()
pred_signals[:, 0::2] = scaler_t0.inverse_transform(pred_signals[:, 0::2])  # t0s
pred_signals[:, 1::2] = scaler_amp.inverse_transform(pred_signals[:, 1::2])  # amps

# ----------------------------
# Prepare comparison arrays
# ----------------------------
pred_t0s_all, true_t0s_all = [], []
pred_amps_all, true_amps_all = [], []

for i in range(len(X)):
    count = min(pred_counts[i], pred_signals.shape[1] // 2)  # Clamp to model output size
    for j in range(count):
        pred_t0 = pred_signals[i][2 * j]
        pred_amp = pred_signals[i][2 * j + 1]
        true_t0, true_amp = y_true[i][j]

        # Only add non-zero predictions
        if pred_t0 != 0 or pred_amp != 0:
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
plt.xlabel("True t₀ (ns)")
plt.ylabel("Predicted t₀ (ns)")
plt.title("Scatter Plot: True vs Predicted t₀")
plt.legend()
plt.grid(True)
plt.savefig("comparison_plots/true_vs_predicted_t0.png")
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(true_amps_all, pred_amps_all, alpha=0.6, edgecolors='k')
plt.plot([min(true_amps_all), max(true_amps_all)], [min(true_amps_all), max(true_amps_all)], 'r--', label="Ideal")
plt.xlabel("True Amplitude")
plt.ylabel("Predicted Amplitude")
plt.title("Scatter Plot: True vs Predicted Amplitude")
plt.legend()
plt.grid(True)
plt.savefig("comparison_plots/true_vs_predicted_amplitude.png")
plt.show()

print("✅ Saved comparison plots in 'comparison_plots/' folder")
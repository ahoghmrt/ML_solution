import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow import keras
import joblib

# @tf.keras.utils.register_keras_serializable()
# def custom_loss(y_true, y_pred):
#     y_true = tf.reshape(y_true, (-1, max_signals, 3))
#     y_pred = tf.reshape(y_pred, (-1, max_signals, 3))

#     t0_true, amp_true, pres_true = tf.split(y_true, 3, axis=-1)
#     t0_pred, amp_pred, pres_pred_raw = tf.split(y_pred, 3, axis=-1)

#     pres_pred = tf.sigmoid(pres_pred_raw)
#     bce = tf.keras.losses.binary_crossentropy(pres_true, pres_pred)
#     presence_mask = pres_true

#     t0_loss = tf.reduce_mean(tf.abs(t0_pred - t0_true) * presence_mask)
#     amp_loss = tf.reduce_mean(tf.abs(amp_pred - amp_true) * presence_mask)
#     presence_loss = tf.reduce_mean(bce)

#     return 0.3 * t0_loss + 0.6 * amp_loss + 0.1 * presence_loss

# # ----------------------------
# # Load model and scalers
# # ----------------------------
# signal_model = keras.models.load_model(
#     "signal_model.keras",
#     custom_objects={"custom_loss": custom_loss}
# )

# ----------------------------
# Load model and scalers
# ----------------------------
signal_model = keras.models.load_model("signal_model.keras")
scaler_wave = joblib.load("training_plots/waveform_scaler.pkl")
scaler_t0 = joblib.load("training_plots/t0_scaler.pkl")
scaler_amp = joblib.load("training_plots/amp_scaler.pkl")

# ----------------------------
# Load data
# ----------------------------
data = np.load("ml_training_data/training_data_signals.npz")
X = data["waveforms"]
y_true = data["labels"]
time = data["time"]
max_signals = y_true.shape[1]

# ----------------------------
# Predict
# ----------------------------
X_scaled = scaler_wave.transform(X)
pred_signals = signal_model.predict(X_scaled[..., np.newaxis])
pred_t0s = scaler_t0.inverse_transform(pred_signals[:, 0::3])
pred_amps = scaler_amp.inverse_transform(pred_signals[:, 1::3])
pred_presence = pred_signals[:, 2::3]
mask = pred_presence > 0.5

# ----------------------------
# Prepare outputs
# ----------------------------
os.makedirs("comparison_plots", exist_ok=True)
colors = plt.cm.tab10.colors
slotwise_true_t0s = [[] for _ in range(max_signals)]
slotwise_pred_t0s = [[] for _ in range(max_signals)]
slotwise_true_amps = [[] for _ in range(max_signals)]
slotwise_pred_amps = [[] for _ in range(max_signals)]
true_t0s_all, pred_t0s_all, delta_t0s = [], [], []
true_amps_all, pred_amps_all, delta_amps = [], [], []

# ----------------------------
# Compare
# ----------------------------
for i in range(len(X)):
    for j in range(max_signals):
        if mask[i][j]:
            pred_t0, pred_amp = pred_t0s[i][j], pred_amps[i][j]
            true_t0, true_amp = y_true[i][j]

            slotwise_true_t0s[j].append(true_t0)
            slotwise_pred_t0s[j].append(pred_t0)
            slotwise_true_amps[j].append(true_amp)
            slotwise_pred_amps[j].append(pred_amp)

            true_t0s_all.append(true_t0)
            pred_t0s_all.append(pred_t0)
            delta_t0s.append(pred_t0 - true_t0)

            true_amps_all.append(true_amp)
            pred_amps_all.append(pred_amp)
            delta_amps.append(pred_amp - true_amp)

# ----------------------------
# Plot helpers
# ----------------------------
def scatter_combined(x_all, y_all, slotwise_x, slotwise_y, label, xlabel, ylabel, file_name):
    fig, ax = plt.subplots(figsize=(8, 6))
    for j in range(max_signals):
        ax.scatter(slotwise_x[j], slotwise_y[j], label=f"Signal {j+1}", alpha=0.6, s=15, color=colors[j % 10])
    ax.plot([min(x_all), max(x_all)], [min(x_all), max(x_all)], 'k--', label="Ideal")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(label)
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"comparison_plots/{file_name}.png")
    plt.close()

def delta_histogram(deltas, title, xlabel, file_name, color):
    plt.figure(figsize=(8, 5))
    plt.hist(deltas, bins=100, alpha=0.75, color=color, edgecolor='black')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"comparison_plots/{file_name}.png")
    plt.close()

# ----------------------------
# Combined scatter + delta plots
# ----------------------------
scatter_combined(true_t0s_all, pred_t0s_all, slotwise_true_t0s, slotwise_pred_t0s,
                 "True vs Predicted t₀", "True t₀ (ns)", "Predicted t₀ (ns)", "t0_comparison_colored_by_slot")

scatter_combined(true_amps_all, pred_amps_all, slotwise_true_amps, slotwise_pred_amps,
                 "True vs Predicted Amplitude", "True Amplitude", "Predicted Amplitude", "amplitude_comparison_colored_by_slot")

delta_histogram(delta_t0s, "Δt₀ = Predicted - True", "Δt₀ (ns)", "delta_t0_histogram", 'skyblue')
delta_histogram(delta_amps, "ΔA = Predicted - True", "ΔAmplitude", "delta_amplitude_histogram", 'salmon')

# ----------------------------
# Individual per-signal plots
# ----------------------------
for j in range(max_signals):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.scatter(slotwise_true_t0s[j], slotwise_pred_t0s[j], alpha=0.6, s=15, color=colors[j % 10])
    if slotwise_true_t0s[j]:
        min_val = min(slotwise_true_t0s[j] + slotwise_pred_t0s[j])
        max_val = max(slotwise_true_t0s[j] + slotwise_pred_t0s[j])
        ax1.plot([min_val, max_val], [min_val, max_val], 'k--')
    ax1.set_title(f"Signal {j+1}: True vs Predicted t₀")
    ax1.set_xlabel("True t₀ (ns)")
    ax1.set_ylabel("Predicted t₀ (ns)")
    ax1.grid(True)

    ax2.scatter(slotwise_true_amps[j], slotwise_pred_amps[j], alpha=0.6, s=15, color=colors[j % 10])
    if slotwise_true_amps[j]:
        min_val = min(slotwise_true_amps[j] + slotwise_pred_amps[j])
        max_val = max(slotwise_true_amps[j] + slotwise_pred_amps[j])
        ax2.plot([min_val, max_val], [min_val, max_val], 'k--')
    ax2.set_title(f"Signal {j+1}: True vs Predicted Amplitude")
    ax2.set_xlabel("True Amplitude")
    ax2.set_ylabel("Predicted Amplitude")
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(f"comparison_plots/signal_{j+1}_comparison.png")
    plt.close()

print("✅ All comparison plots saved using presence flag masking.")

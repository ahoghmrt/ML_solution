import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Load dataset
data = np.load("ml_training_data/training_data_signals.npz")
X = data["waveforms"]
y_true = data["labels"]  # shape: (samples, max_signals, 3)
time = data["time"]
max_signals = y_true.shape[1]

# Normalize waveform
X_mean = X.mean(axis=1, keepdims=True)
X_std = X.std(axis=1, keepdims=True)
X_norm = (X - X_mean) / (X_std + 1e-8)

# Load updated model
model = load_model("ml_training_data/signal_regression_model.keras")

# Predict
pred_outputs = model.predict(X_norm)
y_pred_t0_amp = pred_outputs[0].reshape((-1, max_signals, 2))
y_pred_presence = pred_outputs[1].reshape((-1, max_signals, 1))
y_pred = np.concatenate([y_pred_t0_amp, y_pred_presence], axis=2)

# Show sample predictions
print("\n\U0001F50D Sample Predictions:")
for i in range(5):  # first 5 samples
    print(f"\nSample {i}")
    for j in range(max_signals):
        print(f"  Signal {j+1}: Predicted -> t0 = {y_pred[i][j][0]:.2f} ns, "
              f"A = {y_pred[i][j][1]:.2f}, Presence = {y_pred[i][j][2]:.2f}")
        print(f"              True     -> t0 = {y_true[i][j][0]:.2f} ns, "
              f"A = {y_true[i][j][1]:.2f}, Presence = {y_true[i][j][2]:.2f}")

# Scatter plot of all True vs Predicted t0s and Amplitudes
true_t0s_all = []
pred_t0s_all = []
true_amps_all = []
pred_amps_all = []

for i in range(len(X)):
    for j in range(max_signals):
        if y_true[i][j][2] > 0.5 and y_pred[i][j][2] > 0.5:
            true_t0s_all.append(y_true[i][j][0])
            pred_t0s_all.append(y_pred[i][j][0])
            true_amps_all.append(y_true[i][j][1])
            pred_amps_all.append(y_pred[i][j][1])

# Scatter plots
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
axes[0].scatter(true_t0s_all, pred_t0s_all, alpha=0.6, edgecolors='k')
axes[0].plot([min(true_t0s_all), max(true_t0s_all)], [min(true_t0s_all), max(true_t0s_all)], 'r--')
axes[0].set_title("True t₀ vs Predicted t₀")
axes[0].set_xlabel("True t₀ (ns)")
axes[0].set_ylabel("Predicted t₀ (ns)")
axes[0].grid(True)

axes[1].scatter(true_amps_all, pred_amps_all, alpha=0.6, edgecolors='k')
axes[1].plot([min(true_amps_all), max(true_amps_all)], [min(true_amps_all), max(true_amps_all)], 'r--')
axes[1].set_title("True Amplitude vs Predicted Amplitude")
axes[1].set_xlabel("True Amplitude")
axes[1].set_ylabel("Predicted Amplitude")
axes[1].grid(True)

plt.tight_layout()
plt.savefig("ml_training_data/scatter_true_vs_pred_t0_amp.png")
plt.show()
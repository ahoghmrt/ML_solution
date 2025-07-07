import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow import keras
import joblib

# ----------------------------
# Load models and scalers
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
# Choose range to visualize
# ----------------------------
start_index = 1   # 👈 change here
end_index = 20    # 👈 and here

os.makedirs("waveform_inspection", exist_ok=True)

# Normalize waveforms
X_scaled = scaler_wave.transform(X)

for idx in range(start_index, end_index):
    waveform = X[idx]
    true_signals = y_true[idx]
    pred_signals_norm = signal_model.predict(X_scaled[[idx]][..., np.newaxis])[0]

    # Extract predicted components
    pred_t0_norm = pred_signals_norm[0::3]
    pred_amp_norm = pred_signals_norm[1::3]
    pred_presences = pred_signals_norm[2::3]

    # Apply inverse transforms
    pred_t0 = scaler_t0.inverse_transform([pred_t0_norm])[0]
    pred_amp = scaler_amp.inverse_transform([pred_amp_norm])[0]

    # Use presence flag to mask predictions
    mask = pred_presences > 0.5
    pred_t0 = pred_t0[mask]
    pred_amp = pred_amp[mask]

    # Extract true signals
    true_t0, true_amp = [], []
    for t0, amp in true_signals:
        if t0 > 0 or amp > 0:
            true_t0.append(t0)
            true_amp.append(amp)

    # Plot
    plt.figure(figsize=(10, 4))
    plt.plot(time, waveform, color='gray', label='Waveform')
    plt.scatter(true_t0, true_amp, color='blue', edgecolors='k', label='True', s=60)
    plt.scatter(pred_t0, pred_amp, color='green', edgecolors='k', marker='x', label='Predicted', s=60)
    plt.title(f"Waveform #{idx}")
    plt.xlabel("Time (ns)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)

    fname = f"waveform_inspection/waveform_{idx:03d}.png"
    plt.savefig(fname)
    plt.close()

print(f"✅ Saved {end_index - start_index} waveform plots in 'waveform_inspection/' folder.")
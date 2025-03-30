import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import os
from sklearn.preprocessing import StandardScaler

# ----------------------------
# Load models
# ----------------------------
signal_model = keras.models.load_model("signal_model_using_counts.keras")
count_model = keras.models.load_model("signal_count_model_with_candidates.keras")

# ----------------------------
# Load data
# ----------------------------
data_sig = np.load("ml_training_data/training_data_signals.npz")
data_cnt = np.load("ml_training_data/training_data_counts.npz")

X = data_sig["waveforms"]
y_true = data_sig["labels"]
time = data_sig["time"]
X_t0 = data_cnt["candidate_t0s"]
X_amp = data_cnt["candidate_amps"]

# ----------------------------
# Normalize inputs
# ----------------------------
X_scaled = keras.utils.normalize(X, axis=1)
X_t0_scaled = StandardScaler().fit_transform(X_t0)
X_amp_scaled = StandardScaler().fit_transform(X_amp)

# ----------------------------
# Choose range to visualize
# ----------------------------
start_index = 31
end_index = 60

os.makedirs("waveform_inspection", exist_ok=True)

for idx in range(start_index, end_index):
    waveform = X[idx]
    true_signals = y_true[idx]

    # Prepare model inputs for this waveform
    x_input = X_scaled[[idx]]
    x_t0_input = X_t0_scaled[[idx]]
    x_amp_input = X_amp_scaled[[idx]]

    pred_count = np.argmax(count_model.predict([x_input, x_t0_input, x_amp_input]), axis=1)[0]
    pred_signals = signal_model.predict(X[[idx]])[0]
    pred_count = min(pred_count, pred_signals.shape[0])  # Clamp to max_signals

    # Extract true signals
    true_t0, true_amp = [], []
    for t0, amp in true_signals:
        if t0 > 0 or amp > 0:
            true_t0.append(t0)
            true_amp.append(amp)

    # Extract predicted signals
    pred_t0, pred_amp = [], []
    for j in range(pred_count):
        t0, amp = pred_signals[j]
        pred_t0.append(t0)
        pred_amp.append(amp)

    # Plot
    plt.figure(figsize=(10, 4))
    plt.plot(time, waveform, color='gray', label='Waveform')
    plt.scatter(true_t0, true_amp, color='blue', edgecolors='k', label='True', s=60)
    plt.scatter(pred_t0, pred_amp, color='green', marker='x', label='Predicted', s=60)
    plt.title(f"Waveform #{idx}")
    plt.xlabel("Time (ns)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)

    fname = f"waveform_inspection/waveform_{idx:03d}.png"
    plt.savefig(fname)
    plt.close()

print(f"✅ Saved {end_index - start_index} waveform plots in 'waveform_inspection/' folder.")

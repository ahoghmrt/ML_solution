import numpy as np
import logging
import matplotlib.pyplot as plt
import os
from tensorflow import keras
import joblib

logger = logging.getLogger(__name__)


def main(start=1, end=300):
    # ----------------------------
    # Load models and scalers
    # ----------------------------
    signal_model = keras.models.load_model("signal_model.keras")
    count_model = keras.models.load_model("signal_count_model.keras")
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

    os.makedirs("waveform_inspection", exist_ok=True)

    # Normalize waveforms
    X_scaled = scaler_wave.transform(X)

    for idx in range(start, end):
        waveform = X[idx]
        true_signals = y_true[idx]
        pred_count = np.argmax(count_model.predict(X_scaled[[idx]]), axis=1)[0]
        pred_signals_norm = signal_model.predict(X_scaled[[idx]][..., np.newaxis])[0]

        # Inverse transform predictions
        pred_signals = pred_signals_norm.copy()
        pred_signals[0::2] = scaler_t0.inverse_transform([pred_signals[0::2]])[0]
        pred_signals[1::2] = scaler_amp.inverse_transform([pred_signals[1::2]])[0]

        # Extract true signals
        true_t0, true_amp = [], []
        for t0, amp in true_signals:
            if t0 > 0 or amp > 0:
                true_t0.append(t0)
                true_amp.append(amp)

        # Extract predicted signals
        pred_t0, pred_amp = [], []
        for j in range(pred_count):
            t0 = pred_signals[2 * j]
            amp = pred_signals[2 * j + 1]
            pred_t0.append(t0)
            pred_amp.append(amp)

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

    logger.info(f"Saved {end - start} waveform plots in 'waveform_inspection/' folder.")


if __name__ == "__main__":
    main()

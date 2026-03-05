import numpy as np
import logging
import matplotlib.pyplot as plt
import os
from tensorflow import keras
import joblib
import config as cfg

logger = logging.getLogger(__name__)


def main(start=cfg.PLOT_START, end=cfg.PLOT_END):
    # ----------------------------
    # Load models and scalers
    # ----------------------------
    signal_model = keras.models.load_model("signal_model.keras")
    count_model = keras.models.load_model("signal_count_model.keras")
    scaler_wave = joblib.load(os.path.join(cfg.DIR_TRAINING_PLOTS, "waveform_scaler.pkl"))
    scaler_count_wave = joblib.load(os.path.join(cfg.DIR_TRAINING_PLOTS, "count_waveform_scaler.pkl"))
    scaler_t0 = joblib.load(os.path.join(cfg.DIR_TRAINING_PLOTS, "t0_scaler.pkl"))
    scaler_amp = joblib.load(os.path.join(cfg.DIR_TRAINING_PLOTS, "amp_scaler.pkl"))

    # ----------------------------
    # Load data
    # ----------------------------
    data = np.load(os.path.join(cfg.DIR_ML_DATA, "training_data_signals.npz"))
    X = data["waveforms"]
    y_true = data["labels"]
    time = data["time"]

    os.makedirs(cfg.DIR_WAVEFORM_INSPECTION, exist_ok=True)

    # Normalize waveforms (each model uses its own scaler)
    X_scaled_signal = scaler_wave.transform(X)
    X_scaled_count = scaler_count_wave.transform(X)

    # Batch predict on the full range at once
    batch_counts = np.argmax(
        count_model.predict(X_scaled_count[start:end][..., np.newaxis]), axis=1)
    batch_signals_norm = signal_model.predict(
        X_scaled_signal[start:end][..., np.newaxis])

    # Inverse transform all predictions at once
    batch_signals = batch_signals_norm.copy()
    batch_signals[:, 0::2] = scaler_t0.inverse_transform(batch_signals_norm[:, 0::2])
    batch_signals[:, 1::2] = scaler_amp.inverse_transform(batch_signals_norm[:, 1::2])

    for i, idx in enumerate(range(start, end)):
        waveform = X[idx]
        true_signals = y_true[idx]
        pred_count = batch_counts[i]
        pred_signals = batch_signals[i]

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
        plt.scatter(pred_t0, pred_amp, color='green', marker='x', label='Predicted', s=60)
        plt.title(f"Waveform #{idx}")
        plt.xlabel("Time (ns)")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.grid(True)

        fname = os.path.join(cfg.DIR_WAVEFORM_INSPECTION, f"waveform_{idx:03d}.png")
        plt.savefig(fname)
        plt.close()

    logger.info(f"Saved {end - start} waveform plots to '{cfg.DIR_WAVEFORM_INSPECTION}/'")


if __name__ == "__main__":
    main()

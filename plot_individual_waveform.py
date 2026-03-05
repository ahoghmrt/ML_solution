import numpy as np
import logging
import matplotlib.pyplot as plt
import os
from tensorflow import keras
import joblib
from scipy.optimize import linear_sum_assignment
import config as cfg

logger = logging.getLogger(__name__)


def main(start=cfg.PLOT_START, end=cfg.PLOT_END):
    # ----------------------------
    # Load models and scalers
    # ----------------------------
    from train_signal_model import WeightedHuberLoss  # register custom loss
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

        # Hungarian matching between pred and true signals
        n_pred = len(pred_t0)
        n_true = len(true_t0)
        matches = []
        if n_pred > 0 and n_true > 0:
            pred_arr = np.array(list(zip(pred_t0, pred_amp)))
            true_arr = np.array(list(zip(true_t0, true_amp)))
            cost = np.abs(pred_arr[:, np.newaxis, 0] - true_arr[np.newaxis, :, 0]) + \
                   np.abs(pred_arr[:, np.newaxis, 1] - true_arr[np.newaxis, :, 1])
            row_ind, col_ind = linear_sum_assignment(cost)
            matches = list(zip(row_ind, col_ind))

        # Plot
        plt.figure(figsize=(10, 4))
        plt.plot(time, waveform, color='gray', label='Waveform')
        plt.scatter(true_t0, true_amp, color='blue', edgecolors='k', label='True', s=60)
        plt.scatter(pred_t0, pred_amp, color='green', marker='x', label='Predicted', s=60)
        for pi, ti in matches:
            plt.plot([pred_t0[pi], true_t0[ti]], [pred_amp[pi], true_amp[ti]],
                     'k--', alpha=0.4, linewidth=0.8)
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

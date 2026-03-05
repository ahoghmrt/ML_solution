import numpy as np
import logging
import matplotlib.pyplot as plt
import os
from tensorflow import keras
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import confusion_matrix, accuracy_score
from scipy.stats import pearsonr, spearmanr

logger = logging.getLogger(__name__)


def main():
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

    count_data = np.load("ml_training_data/training_data_counts.npz")
    y_true_counts = count_data["labels"]  # (samples,)

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
    delta_t0s, delta_amps = [], []
    sample_true_counts = []

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
                delta_t0s.append(pred_t0 - true_t0)
                delta_amps.append(pred_amp - true_amp)
                sample_true_counts.append(y_true_counts[i])

    # ----------------------------
    # Numeric Metrics
    # ----------------------------
    true_t0s = np.array(true_t0s_all)
    pred_t0s = np.array(pred_t0s_all)
    true_amps = np.array(true_amps_all)
    pred_amps = np.array(pred_amps_all)
    dt0 = np.array(delta_t0s)
    damp = np.array(delta_amps)
    counts_arr = np.array(sample_true_counts)

    t0_mae = mean_absolute_error(true_t0s, pred_t0s)
    t0_rmse = np.sqrt(mean_squared_error(true_t0s, pred_t0s))
    t0_r2 = r2_score(true_t0s, pred_t0s)
    t0_pearson, _ = pearsonr(true_t0s, pred_t0s)
    t0_spearman, _ = spearmanr(true_t0s, pred_t0s)

    amp_mae = mean_absolute_error(true_amps, pred_amps)
    amp_rmse = np.sqrt(mean_squared_error(true_amps, pred_amps))
    amp_r2 = r2_score(true_amps, pred_amps)
    amp_pearson, _ = pearsonr(true_amps, pred_amps)
    amp_spearman, _ = spearmanr(true_amps, pred_amps)

    count_accuracy = accuracy_score(y_true_counts, pred_counts)

    logger.info("=" * 50)
    logger.info("Signal Prediction Metrics (positional matching)")
    logger.info("=" * 50)
    logger.info(f"  t0  - MAE: {t0_mae:.4f} ns, RMSE: {t0_rmse:.4f} ns, R2: {t0_r2:.4f}, "
                f"Pearson: {t0_pearson:.4f}, Spearman: {t0_spearman:.4f}")
    logger.info(f"  amp - MAE: {amp_mae:.4f}, RMSE: {amp_rmse:.4f}, R2: {amp_r2:.4f}, "
                f"Pearson: {amp_pearson:.4f}, Spearman: {amp_spearman:.4f}")
    logger.info(f"  Count model accuracy: {count_accuracy:.4f}")

    logger.info("  Per-true-count breakdown:")
    for c in sorted(np.unique(counts_arr)):
        mask = counts_arr == c
        c_t0_mae = mean_absolute_error(true_t0s[mask], pred_t0s[mask])
        c_amp_mae = mean_absolute_error(true_amps[mask], pred_amps[mask])
        logger.info(f"    count={int(c)}: n={mask.sum()}, t0 MAE={c_t0_mae:.4f} ns, amp MAE={c_amp_mae:.4f}")
    logger.info("=" * 50)

    metrics = {
        't0_mae': float(t0_mae), 't0_rmse': float(t0_rmse), 't0_r2': float(t0_r2),
        't0_pearson': float(t0_pearson), 't0_spearman': float(t0_spearman),
        'amp_mae': float(amp_mae), 'amp_rmse': float(amp_rmse), 'amp_r2': float(amp_r2),
        'amp_pearson': float(amp_pearson), 'amp_spearman': float(amp_spearman),
        'count_accuracy': float(count_accuracy),
        'n_signal_pairs': len(pred_t0s_all),
    }

    # ----------------------------
    # Plot scatter comparisons
    # ----------------------------
    os.makedirs("comparison_plots", exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Hexbin plot for t₀
    hb1 = ax1.hexbin(true_t0s_all, pred_t0s_all, gridsize=80, cmap='viridis', mincnt=1)
    ax1.plot([min(true_t0s_all), max(true_t0s_all)],
             [min(true_t0s_all), max(true_t0s_all)], 'r--', label="Ideal")
    ax1.set_xlabel("True t₀ (ns)")
    ax1.set_ylabel("Predicted t₀ (ns)")
    ax1.set_title("Hexbin: True vs Predicted t₀")
    ax1.grid(True)
    plt.colorbar(hb1, ax=ax1, label='Counts')
    ax1.legend()

    # Right: Δt₀ histogram
    ax2.hist(delta_t0s, bins=100, alpha=0.75, color='skyblue', edgecolor='black')
    ax2.set_title("Δt₀ = Predicted - True")
    ax2.set_xlabel("Δt₀ (ns)")
    ax2.set_ylabel("Count")
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig("comparison_plots/t0_comparison_combined.png")
    plt.close()

    # ----------------------------------------------

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Hexbin plot for amplitude
    hb2 = ax1.hexbin(true_amps_all, pred_amps_all, gridsize=80, cmap='plasma', mincnt=1)
    ax1.plot([min(true_amps_all), max(true_amps_all)],
             [min(true_amps_all), max(true_amps_all)], 'r--', label="Ideal")
    ax1.set_xlabel("True Amplitude")
    ax1.set_ylabel("Predicted Amplitude")
    ax1.set_title("Hexbin: True vs Predicted Amplitude")
    ax1.grid(True)
    plt.colorbar(hb2, ax=ax1, label='Counts')
    ax1.legend()

    # Right: ΔAmplitude histogram
    ax2.hist(delta_amps, bins=100, alpha=0.75, color='salmon', edgecolor='black')
    ax2.set_title("ΔA = Predicted - True")
    ax2.set_xlabel("ΔAmplitude")
    ax2.set_ylabel("Count")
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig("comparison_plots/amplitude_comparison_combined.png")
    plt.close()

    # ----------------------------
    # Count Model Confusion Matrix
    # ----------------------------
    cm = confusion_matrix(y_true_counts, pred_counts, labels=np.arange(7))
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.set_title("Count Model Confusion Matrix")
    ax.set_xlabel("Predicted Count")
    ax.set_ylabel("True Count")
    ax.set_xticks(np.arange(7))
    ax.set_yticks(np.arange(7))
    plt.colorbar(im, ax=ax)
    for i in range(7):
        for j in range(7):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] > cm.max() / 2 else 'black')
    plt.tight_layout()
    plt.savefig("comparison_plots/count_confusion_matrix.png")
    plt.close()

    logger.info(f"Comparison plots saved ({len(pred_t0s_all)} signal pairs, "
                f"count accuracy: {count_accuracy:.4f})")

    return metrics


if __name__ == "__main__":
    main()

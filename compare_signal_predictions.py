import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow import keras
import joblib


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
    if os.environ.get("DISPLAY"):
        plt.show()
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
    if os.environ.get("DISPLAY"):
        plt.show()
    plt.close()

    print("✅ Hexbin plots and delta histograms saved in 'comparison_plots/' folder.")


if __name__ == "__main__":
    main()

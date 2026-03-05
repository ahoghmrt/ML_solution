"""Error analysis module for ADC waveform signal extraction experiments.

Takes an experiment folder as input, loads models/data, and produces
detailed error diagnostics with plots and a JSON report.
"""

import json
import logging
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
)
from tensorflow import keras

import config as cfg

logger = logging.getLogger(__name__)


def _load_predictions(experiment_dir):
    """Load models, scalers, data, and compute all predictions."""
    # Load config from experiment
    with open(os.path.join(experiment_dir, "config.json")) as f:
        exp_config = json.load(f)

    # Load models and scalers
    signal_model = keras.models.load_model("signal_model.keras")
    count_model = keras.models.load_model("signal_count_model.keras")
    scaler_wave = joblib.load(os.path.join(cfg.DIR_TRAINING_PLOTS, "waveform_scaler.pkl"))
    scaler_count_wave = joblib.load(os.path.join(cfg.DIR_TRAINING_PLOTS, "count_waveform_scaler.pkl"))
    scaler_t0 = joblib.load(os.path.join(cfg.DIR_TRAINING_PLOTS, "t0_scaler.pkl"))
    scaler_amp = joblib.load(os.path.join(cfg.DIR_TRAINING_PLOTS, "amp_scaler.pkl"))

    # Load data
    sig_data = np.load(os.path.join(cfg.DIR_ML_DATA, "training_data_signals.npz"))
    X = sig_data["waveforms"]
    y_true = sig_data["labels"]  # (N, max_signals, 2)
    time = sig_data["time"]

    cnt_data = np.load(os.path.join(cfg.DIR_ML_DATA, "training_data_counts.npz"))
    y_true_counts = cnt_data["labels"]  # (N,)

    # Normalize and predict
    X_sig = scaler_wave.transform(X)[..., np.newaxis]
    X_cnt = scaler_count_wave.transform(X)[..., np.newaxis]

    pred_counts = np.argmax(count_model.predict(X_cnt), axis=1)
    pred_signals_norm = signal_model.predict(X_sig)

    # Inverse transform
    pred_signals = pred_signals_norm.copy()
    pred_signals[:, 0::2] = scaler_t0.inverse_transform(pred_signals_norm[:, 0::2])
    pred_signals[:, 1::2] = scaler_amp.inverse_transform(pred_signals_norm[:, 1::2])

    return {
        "X": X, "time": time,
        "y_true": y_true, "y_true_counts": y_true_counts,
        "pred_signals": pred_signals, "pred_counts": pred_counts,
        "exp_config": exp_config,
    }


def _per_slot_analysis(data, out_dir):
    """Per-slot MAE and RMSE for t0 and amplitude."""
    y_true = data["y_true"]
    pred = data["pred_signals"]
    max_signals = y_true.shape[1]

    slot_t0_mae, slot_amp_mae = [], []
    slot_t0_rmse, slot_amp_rmse = [], []

    for s in range(max_signals):
        true_t0 = y_true[:, s, 0]
        true_amp = y_true[:, s, 1]
        pred_t0 = pred[:, 2 * s]
        pred_amp = pred[:, 2 * s + 1]

        # Only evaluate on samples that actually have this slot filled
        active = (true_t0 > 0) | (true_amp > 0)
        if active.sum() == 0:
            slot_t0_mae.append(0)
            slot_amp_mae.append(0)
            slot_t0_rmse.append(0)
            slot_amp_rmse.append(0)
            continue

        slot_t0_mae.append(mean_absolute_error(true_t0[active], pred_t0[active]))
        slot_amp_mae.append(mean_absolute_error(true_amp[active], pred_amp[active]))
        slot_t0_rmse.append(np.sqrt(mean_squared_error(true_t0[active], pred_t0[active])))
        slot_amp_rmse.append(np.sqrt(mean_squared_error(true_amp[active], pred_amp[active])))

    slots = np.arange(max_signals)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].bar(slots, slot_t0_mae, color="steelblue", edgecolor="black")
    axes[0, 0].set_title("Per-Slot t0 MAE")
    axes[0, 0].set_xlabel("Signal Slot")
    axes[0, 0].set_ylabel("MAE (ns)")
    axes[0, 0].set_xticks(slots)

    axes[0, 1].bar(slots, slot_amp_mae, color="salmon", edgecolor="black")
    axes[0, 1].set_title("Per-Slot Amplitude MAE")
    axes[0, 1].set_xlabel("Signal Slot")
    axes[0, 1].set_ylabel("MAE")
    axes[0, 1].set_xticks(slots)

    axes[1, 0].bar(slots, slot_t0_rmse, color="steelblue", edgecolor="black", alpha=0.7)
    axes[1, 0].set_title("Per-Slot t0 RMSE")
    axes[1, 0].set_xlabel("Signal Slot")
    axes[1, 0].set_ylabel("RMSE (ns)")
    axes[1, 0].set_xticks(slots)

    axes[1, 1].bar(slots, slot_amp_rmse, color="salmon", edgecolor="black", alpha=0.7)
    axes[1, 1].set_title("Per-Slot Amplitude RMSE")
    axes[1, 1].set_xlabel("Signal Slot")
    axes[1, 1].set_ylabel("RMSE")
    axes[1, 1].set_xticks(slots)

    for ax in axes.flat:
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "per_slot_mae.png"), dpi=150)
    plt.close()

    return {
        "slot_t0_mae": [float(v) for v in slot_t0_mae],
        "slot_amp_mae": [float(v) for v in slot_amp_mae],
        "slot_t0_rmse": [float(v) for v in slot_t0_rmse],
        "slot_amp_rmse": [float(v) for v in slot_amp_rmse],
    }


def _error_vs_count(data, out_dir):
    """Box plots of errors grouped by true signal count."""
    y_true = data["y_true"]
    pred = data["pred_signals"]
    pred_counts = data["pred_counts"]
    y_true_counts = data["y_true_counts"]

    # Collect per-signal errors grouped by true count
    count_groups_t0 = {c: [] for c in sorted(np.unique(y_true_counts)) if c > 0}
    count_groups_amp = {c: [] for c in sorted(np.unique(y_true_counts)) if c > 0}

    for i in range(len(y_true)):
        tc = y_true_counts[i]
        if tc == 0:
            continue
        n = min(int(pred_counts[i]), int(tc))
        for j in range(n):
            dt0 = pred[i, 2 * j] - y_true[i, j, 0]
            damp = pred[i, 2 * j + 1] - y_true[i, j, 1]
            count_groups_t0[tc].append(dt0)
            count_groups_amp[tc].append(damp)

    counts = sorted(count_groups_t0.keys())
    t0_data = [count_groups_t0[c] for c in counts]
    amp_data = [count_groups_amp[c] for c in counts]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    bp1 = ax1.boxplot(t0_data, labels=[str(c) for c in counts], patch_artist=True,
                      showfliers=False, medianprops=dict(color="red"))
    for patch in bp1["boxes"]:
        patch.set_facecolor("steelblue")
        patch.set_alpha(0.6)
    ax1.set_title("t0 Error Distribution by Signal Count")
    ax1.set_xlabel("True Signal Count")
    ax1.set_ylabel("Predicted - True t0 (ns)")
    ax1.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax1.grid(axis="y", alpha=0.3)

    bp2 = ax2.boxplot(amp_data, labels=[str(c) for c in counts], patch_artist=True,
                      showfliers=False, medianprops=dict(color="red"))
    for patch in bp2["boxes"]:
        patch.set_facecolor("salmon")
        patch.set_alpha(0.6)
    ax2.set_title("Amplitude Error Distribution by Signal Count")
    ax2.set_xlabel("True Signal Count")
    ax2.set_ylabel("Predicted - True Amplitude")
    ax2.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "error_vs_count_boxplot.png"), dpi=150)
    plt.close()

    return {
        "per_count_t0_bias": {int(c): float(np.mean(count_groups_t0[c])) for c in counts},
        "per_count_amp_bias": {int(c): float(np.mean(count_groups_amp[c])) for c in counts},
        "per_count_t0_std": {int(c): float(np.std(count_groups_t0[c])) for c in counts},
        "per_count_amp_std": {int(c): float(np.std(count_groups_amp[c])) for c in counts},
    }


def _temporal_error_profile(data, out_dir):
    """Error as a function of true t0 position in the time window."""
    y_true = data["y_true"]
    pred = data["pred_signals"]
    pred_counts = data["pred_counts"]
    y_true_counts = data["y_true_counts"]

    true_t0s, errors_t0, errors_amp = [], [], []
    for i in range(len(y_true)):
        n = min(int(pred_counts[i]), int(y_true_counts[i]))
        for j in range(n):
            tt0 = y_true[i, j, 0]
            if tt0 <= 0:
                continue
            true_t0s.append(tt0)
            errors_t0.append(abs(pred[i, 2 * j] - tt0))
            errors_amp.append(abs(pred[i, 2 * j + 1] - y_true[i, j, 1]))

    true_t0s = np.array(true_t0s)
    errors_t0 = np.array(errors_t0)
    errors_amp = np.array(errors_amp)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    hb1 = ax1.hexbin(true_t0s, errors_t0, gridsize=50, cmap="viridis", mincnt=1)
    ax1.set_title("t0 Absolute Error vs True t0 Position")
    ax1.set_xlabel("True t0 (ns)")
    ax1.set_ylabel("|t0 error| (ns)")
    plt.colorbar(hb1, ax=ax1, label="Counts")

    hb2 = ax2.hexbin(true_t0s, errors_amp, gridsize=50, cmap="plasma", mincnt=1)
    ax2.set_title("Amplitude Absolute Error vs True t0 Position")
    ax2.set_xlabel("True t0 (ns)")
    ax2.set_ylabel("|Amplitude error|")
    plt.colorbar(hb2, ax=ax2, label="Counts")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "temporal_error_profile.png"), dpi=150)
    plt.close()

    # Binned statistics
    bins = np.linspace(0, cfg.TIME_END, 13)
    bin_idx = np.digitize(true_t0s, bins) - 1
    binned_t0_mae = {}
    for b in range(len(bins) - 1):
        mask = bin_idx == b
        if mask.sum() > 0:
            label = f"{bins[b]:.0f}-{bins[b+1]:.0f}ns"
            binned_t0_mae[label] = float(np.mean(errors_t0[mask]))

    return {"binned_t0_mae_by_position": binned_t0_mae}


def _amplitude_error_analysis(data, out_dir):
    """Error as a function of true signal amplitude."""
    y_true = data["y_true"]
    pred = data["pred_signals"]
    pred_counts = data["pred_counts"]
    y_true_counts = data["y_true_counts"]

    true_amps, errors_amp, errors_t0 = [], [], []
    for i in range(len(y_true)):
        n = min(int(pred_counts[i]), int(y_true_counts[i]))
        for j in range(n):
            ta = y_true[i, j, 1]
            if ta <= 0:
                continue
            true_amps.append(ta)
            errors_amp.append(abs(pred[i, 2 * j + 1] - ta))
            errors_t0.append(abs(pred[i, 2 * j] - y_true[i, j, 0]))

    true_amps = np.array(true_amps)
    errors_amp = np.array(errors_amp)
    errors_t0 = np.array(errors_t0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    hb1 = ax1.hexbin(true_amps, errors_amp, gridsize=50, cmap="plasma", mincnt=1)
    ax1.set_title("Amplitude Error vs True Amplitude")
    ax1.set_xlabel("True Amplitude")
    ax1.set_ylabel("|Amplitude error|")
    plt.colorbar(hb1, ax=ax1, label="Counts")

    hb2 = ax2.hexbin(true_amps, errors_t0, gridsize=50, cmap="viridis", mincnt=1)
    ax2.set_title("t0 Error vs True Amplitude")
    ax2.set_xlabel("True Amplitude")
    ax2.set_ylabel("|t0 error| (ns)")
    plt.colorbar(hb2, ax=ax2, label="Counts")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "amplitude_error_scatter.png"), dpi=150)
    plt.close()

    # Binned by amplitude quartiles
    quartiles = np.percentile(true_amps, [25, 50, 75])
    labels = [
        f"<{quartiles[0]:.1f}",
        f"{quartiles[0]:.1f}-{quartiles[1]:.1f}",
        f"{quartiles[1]:.1f}-{quartiles[2]:.1f}",
        f">{quartiles[2]:.1f}",
    ]
    bins = [0, quartiles[0], quartiles[1], quartiles[2], np.inf]
    bin_idx = np.digitize(true_amps, bins) - 1
    quartile_metrics = {}
    for b, label in enumerate(labels):
        mask = bin_idx == b
        if mask.sum() > 0:
            quartile_metrics[label] = {
                "amp_mae": float(np.mean(errors_amp[mask])),
                "t0_mae": float(np.mean(errors_t0[mask])),
                "n": int(mask.sum()),
            }

    return {"amplitude_quartile_metrics": quartile_metrics}


def _spacing_analysis(data, out_dir):
    """Error vs minimum inter-signal spacing for multi-signal waveforms."""
    y_true = data["y_true"]
    pred = data["pred_signals"]
    pred_counts = data["pred_counts"]
    y_true_counts = data["y_true_counts"]

    min_spacings, waveform_t0_mae, waveform_amp_mae = [], [], []

    for i in range(len(y_true)):
        tc = int(y_true_counts[i])
        if tc < 2:
            continue
        # True t0s for this waveform
        t0s = y_true[i, :tc, 0]
        t0s_sorted = np.sort(t0s)
        min_sp = np.min(np.diff(t0s_sorted))

        n = min(int(pred_counts[i]), tc)
        if n == 0:
            continue
        t0_errs = [abs(pred[i, 2 * j] - y_true[i, j, 0]) for j in range(n)]
        amp_errs = [abs(pred[i, 2 * j + 1] - y_true[i, j, 1]) for j in range(n)]

        min_spacings.append(min_sp)
        waveform_t0_mae.append(np.mean(t0_errs))
        waveform_amp_mae.append(np.mean(amp_errs))

    min_spacings = np.array(min_spacings)
    waveform_t0_mae = np.array(waveform_t0_mae)
    waveform_amp_mae = np.array(waveform_amp_mae)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    hb1 = ax1.hexbin(min_spacings, waveform_t0_mae, gridsize=50, cmap="viridis", mincnt=1)
    ax1.set_title("Waveform t0 MAE vs Min Signal Spacing")
    ax1.set_xlabel("Min spacing between signals (ns)")
    ax1.set_ylabel("Mean |t0 error| (ns)")
    plt.colorbar(hb1, ax=ax1, label="Waveforms")

    hb2 = ax2.hexbin(min_spacings, waveform_amp_mae, gridsize=50, cmap="plasma", mincnt=1)
    ax2.set_title("Waveform Amplitude MAE vs Min Signal Spacing")
    ax2.set_xlabel("Min spacing between signals (ns)")
    ax2.set_ylabel("Mean |Amplitude error|")
    plt.colorbar(hb2, ax=ax2, label="Waveforms")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "spacing_vs_error.png"), dpi=150)
    plt.close()

    # Correlation
    corr_t0, _ = stats.spearmanr(min_spacings, waveform_t0_mae)
    corr_amp, _ = stats.spearmanr(min_spacings, waveform_amp_mae)

    return {
        "spacing_t0_mae_spearman": float(corr_t0),
        "spacing_amp_mae_spearman": float(corr_amp),
        "n_multi_signal_waveforms": len(min_spacings),
    }


def _worst_cases(data, out_dir, top_n=20):
    """Identify and plot the worst-predicted waveforms."""
    y_true = data["y_true"]
    pred = data["pred_signals"]
    pred_counts = data["pred_counts"]
    y_true_counts = data["y_true_counts"]
    X = data["X"]
    time = data["time"]

    # Per-waveform total error
    waveform_errors = np.zeros(len(X))
    for i in range(len(X)):
        tc = int(y_true_counts[i])
        n = min(int(pred_counts[i]), tc)
        if n == 0:
            # Count error only
            waveform_errors[i] = abs(int(pred_counts[i]) - tc) * 10  # penalty
            continue
        for j in range(n):
            waveform_errors[i] += abs(pred[i, 2 * j] - y_true[i, j, 0])
            waveform_errors[i] += abs(pred[i, 2 * j + 1] - y_true[i, j, 1])
        # Add penalty for count mismatch
        waveform_errors[i] += abs(int(pred_counts[i]) - tc) * 10

    worst_idx = np.argsort(waveform_errors)[-top_n:][::-1]

    # Plot worst cases (2x5 grid)
    n_cols, n_rows = 5, min(4, (top_n + 4) // 5)
    n_plot = n_cols * n_rows
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))
    axes = axes.flatten()

    worst_info = []
    for plot_i, idx in enumerate(worst_idx[:n_plot]):
        ax = axes[plot_i]
        ax.plot(time, X[idx], color="gray", linewidth=0.8)

        tc = int(y_true_counts[idx])
        pc = int(pred_counts[idx])

        # True signals
        for j in range(tc):
            t0, amp = y_true[idx, j]
            if t0 > 0 or amp > 0:
                ax.scatter(t0, amp, color="blue", edgecolors="k", s=40, zorder=5)

        # Predicted signals
        for j in range(pc):
            pt0 = pred[idx, 2 * j]
            pamp = pred[idx, 2 * j + 1]
            ax.scatter(pt0, pamp, color="green", marker="x", s=40, zorder=5)

        ax.set_title(f"#{idx} (err={waveform_errors[idx]:.1f})\ntrue={tc} pred={pc}",
                     fontsize=8)
        ax.tick_params(labelsize=7)

        worst_info.append({
            "index": int(idx),
            "total_error": float(waveform_errors[idx]),
            "true_count": tc,
            "pred_count": pc,
        })

    for i in range(len(worst_idx[:n_plot]), len(axes)):
        axes[i].set_visible(False)

    plt.suptitle("Worst-Predicted Waveforms", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "worst_cases.png"), dpi=150, bbox_inches="tight")
    plt.close()

    return {
        "worst_cases": worst_info,
        "error_percentiles": {
            "p50": float(np.percentile(waveform_errors, 50)),
            "p90": float(np.percentile(waveform_errors, 90)),
            "p95": float(np.percentile(waveform_errors, 95)),
            "p99": float(np.percentile(waveform_errors, 99)),
        },
    }


def _count_model_analysis(data, out_dir):
    """Detailed count model error analysis."""
    y_true_counts = data["y_true_counts"]
    pred_counts = data["pred_counts"]

    num_classes = len(cfg.SIGNAL_COUNTS)
    classes = np.arange(num_classes)

    cm = confusion_matrix(y_true_counts, pred_counts, labels=classes)
    report = classification_report(y_true_counts, pred_counts, labels=classes,
                                   output_dict=True, zero_division=0)

    # Most common misclassification pairs
    misclass = []
    for i in range(num_classes):
        for j in range(num_classes):
            if i != j and cm[i, j] > 0:
                misclass.append((i, j, int(cm[i, j])))
    misclass.sort(key=lambda x: -x[2])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Confusion matrix (normalized)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)
    im = axes[0].imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    axes[0].set_title("Normalized Confusion Matrix")
    axes[0].set_xlabel("Predicted Count")
    axes[0].set_ylabel("True Count")
    axes[0].set_xticks(classes)
    axes[0].set_yticks(classes)
    plt.colorbar(im, ax=axes[0], label="Rate")
    for i in range(num_classes):
        for j in range(num_classes):
            axes[0].text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center",
                         color="white" if cm_norm[i, j] > 0.5 else "black", fontsize=7)

    # Per-class accuracy
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    per_class_acc = np.nan_to_num(per_class_acc)
    axes[1].bar(classes, per_class_acc, color="steelblue", edgecolor="black")
    axes[1].set_title("Per-Class Accuracy")
    axes[1].set_xlabel("True Count")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_xticks(classes)
    axes[1].set_ylim(0, 1.05)
    axes[1].grid(axis="y", alpha=0.3)

    # Count error distribution (predicted - true)
    count_errors = pred_counts.astype(int) - y_true_counts.astype(int)
    unique_errs, err_counts = np.unique(count_errors, return_counts=True)
    axes[2].bar(unique_errs, err_counts, color="coral", edgecolor="black")
    axes[2].set_title("Count Prediction Error Distribution")
    axes[2].set_xlabel("Predicted - True Count")
    axes[2].set_ylabel("Frequency")
    axes[2].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "count_model_analysis.png"), dpi=150)
    plt.close()

    return {
        "per_class_accuracy": {int(c): float(a) for c, a in zip(classes, per_class_acc)},
        "top_misclassifications": [
            {"true": int(t), "pred": int(p), "count": c}
            for t, p, c in misclass[:10]
        ],
        "count_error_mean": float(np.mean(count_errors)),
        "count_error_std": float(np.std(count_errors)),
        "classification_report": {
            k: v for k, v in report.items()
            if k not in ("accuracy",)
        },
    }


def _residual_analysis(data, out_dir):
    """Residual statistics and QQ plots."""
    y_true = data["y_true"]
    pred = data["pred_signals"]
    pred_counts = data["pred_counts"]
    y_true_counts = data["y_true_counts"]

    residuals_t0, residuals_amp = [], []
    for i in range(len(y_true)):
        n = min(int(pred_counts[i]), int(y_true_counts[i]))
        for j in range(n):
            if y_true[i, j, 0] > 0 or y_true[i, j, 1] > 0:
                residuals_t0.append(pred[i, 2 * j] - y_true[i, j, 0])
                residuals_amp.append(pred[i, 2 * j + 1] - y_true[i, j, 1])

    residuals_t0 = np.array(residuals_t0)
    residuals_amp = np.array(residuals_amp)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # QQ plots
    stats.probplot(residuals_t0, dist="norm", plot=axes[0, 0])
    axes[0, 0].set_title("t0 Residual QQ Plot")
    axes[0, 0].grid(alpha=0.3)

    stats.probplot(residuals_amp, dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title("Amplitude Residual QQ Plot")
    axes[0, 1].grid(alpha=0.3)

    # Residual histograms with fitted normal
    axes[1, 0].hist(residuals_t0, bins=100, density=True, alpha=0.7, color="steelblue",
                    edgecolor="black", label="Residuals")
    x_t0 = np.linspace(residuals_t0.min(), residuals_t0.max(), 200)
    axes[1, 0].plot(x_t0, stats.norm.pdf(x_t0, residuals_t0.mean(), residuals_t0.std()),
                    "r-", linewidth=2, label="Normal fit")
    axes[1, 0].set_title("t0 Residual Distribution")
    axes[1, 0].set_xlabel("Predicted - True t0 (ns)")
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].hist(residuals_amp, bins=100, density=True, alpha=0.7, color="salmon",
                    edgecolor="black", label="Residuals")
    x_amp = np.linspace(residuals_amp.min(), residuals_amp.max(), 200)
    axes[1, 1].plot(x_amp, stats.norm.pdf(x_amp, residuals_amp.mean(), residuals_amp.std()),
                    "r-", linewidth=2, label="Normal fit")
    axes[1, 1].set_title("Amplitude Residual Distribution")
    axes[1, 1].set_xlabel("Predicted - True Amplitude")
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "residual_qq.png"), dpi=150)
    plt.close()

    # Normality tests (on subsample for large datasets)
    subsample = min(5000, len(residuals_t0))
    idx = np.random.choice(len(residuals_t0), subsample, replace=False)
    _, shapiro_t0_p = stats.shapiro(residuals_t0[idx])
    _, shapiro_amp_p = stats.shapiro(residuals_amp[idx])

    return {
        "t0_residual": {
            "mean": float(residuals_t0.mean()),
            "std": float(residuals_t0.std()),
            "skewness": float(stats.skew(residuals_t0)),
            "kurtosis": float(stats.kurtosis(residuals_t0)),
            "shapiro_p_value": float(shapiro_t0_p),
        },
        "amp_residual": {
            "mean": float(residuals_amp.mean()),
            "std": float(residuals_amp.std()),
            "skewness": float(stats.skew(residuals_amp)),
            "kurtosis": float(stats.kurtosis(residuals_amp)),
            "shapiro_p_value": float(shapiro_amp_p),
        },
        "n_signal_pairs": len(residuals_t0),
    }


def _compile_summary_pdf(out_dir):
    """Combine all individual plots into a single multi-page PDF."""
    plot_files = [
        "per_slot_mae.png",
        "error_vs_count_boxplot.png",
        "temporal_error_profile.png",
        "amplitude_error_scatter.png",
        "spacing_vs_error.png",
        "worst_cases.png",
        "count_model_analysis.png",
        "residual_qq.png",
    ]

    pdf_path = os.path.join(out_dir, "summary.pdf")
    with PdfPages(pdf_path) as pdf:
        for fname in plot_files:
            fpath = os.path.join(out_dir, fname)
            if os.path.isfile(fpath):
                img = plt.imread(fpath)
                fig, ax = plt.subplots(figsize=(16, 10))
                ax.imshow(img)
                ax.axis("off")
                ax.set_title(fname.replace(".png", "").replace("_", " ").title(),
                             fontsize=14, pad=10)
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

    logger.info(f"Summary PDF saved to '{pdf_path}'")


def main(experiment_dir):
    """Run all error analyses on the given experiment folder."""
    out_dir = os.path.join(experiment_dir, "error_analysis")
    os.makedirs(out_dir, exist_ok=True)

    logger.info(f"Running error analysis on '{experiment_dir}'")

    # Load all predictions
    data = _load_predictions(experiment_dir)

    # Run all analyses
    report = {}
    report["per_slot"] = _per_slot_analysis(data, out_dir)
    logger.info("Completed per-slot analysis")

    report["error_vs_count"] = _error_vs_count(data, out_dir)
    logger.info("Completed error vs count analysis")

    report["temporal_profile"] = _temporal_error_profile(data, out_dir)
    logger.info("Completed temporal error profile")

    report["amplitude_dependency"] = _amplitude_error_analysis(data, out_dir)
    logger.info("Completed amplitude error analysis")

    report["spacing"] = _spacing_analysis(data, out_dir)
    logger.info("Completed spacing analysis")

    report["worst_cases"] = _worst_cases(data, out_dir)
    logger.info("Completed worst-case analysis")

    report["count_model"] = _count_model_analysis(data, out_dir)
    logger.info("Completed count model analysis")

    report["residuals"] = _residual_analysis(data, out_dir)
    logger.info("Completed residual analysis")

    # Save JSON report
    report_path = os.path.join(out_dir, "error_analysis_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Saved report to '{report_path}'")

    # Compile all plots into summary PDF
    _compile_summary_pdf(out_dir)

    logger.info(f"Error analysis complete: {out_dir}/")
    return report


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) < 2:
        print("Usage: python -m error_analysis.analyze <experiment_folder>")
        sys.exit(1)
    main(sys.argv[1])

"""Compare the trained ML signal model against classical baselines.

Runs on the same train/val split used for training (via RANDOM_STATE) so
the ML numbers here match the training report. Classical methods are
zero-shot and don't care about the split, but we restrict them to the
validation set too for an apples-to-apples comparison.
"""
from __future__ import annotations
import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from sklearn.model_selection import train_test_split

import config as cfg
from classical_baselines import matched_filter, iterative_clean

logger = logging.getLogger(__name__)


def _match_and_score(
    true_pairs: list[tuple[float, float]],
    pred_pairs: list[tuple[float, float]],
) -> tuple[list[float], list[float], int, int]:
    """Hungarian-match predictions to truth and return per-match errors."""
    if not true_pairs and not pred_pairs:
        return [], [], 0, 0
    if not pred_pairs:
        return [], [], len(true_pairs), 0
    if not true_pairs:
        return [], [], 0, len(pred_pairs)

    true_arr = np.asarray(true_pairs, dtype=float)
    pred_arr = np.asarray(pred_pairs, dtype=float)

    # Normalized cost (each component in its own [0, 1] range)
    cost = (
        np.abs(pred_arr[:, None, 0] - true_arr[None, :, 0]) / cfg.TIME_END
        + np.abs(pred_arr[:, None, 1] - true_arr[None, :, 1]) / cfg.AMPLITUDE_MAX
    )
    row, col = linear_sum_assignment(cost)

    t0_err = [pred_arr[r, 0] - true_arr[c, 0] for r, c in zip(row, col)]
    amp_err = [pred_arr[r, 1] - true_arr[c, 1] for r, c in zip(row, col)]
    n_matched = len(row)
    missed = len(true_pairs) - n_matched
    spurious = len(pred_pairs) - n_matched
    return t0_err, amp_err, missed, spurious


def _ml_predict(waveforms: np.ndarray) -> list[list[tuple[float, float]]]:
    """Run the trained ML models on a batch of waveforms."""
    import joblib
    from tensorflow import keras
    # Register custom loss so .keras loads cleanly
    from train_signal_model import WeightedHuberLoss  # noqa: F401

    signal_model = keras.models.load_model("signal_model.keras")
    count_model = keras.models.load_model("signal_count_model.keras")
    scaler_wave = joblib.load(os.path.join(cfg.DIR_TRAINING_PLOTS, "waveform_scaler.pkl"))
    scaler_cwave = joblib.load(os.path.join(cfg.DIR_TRAINING_PLOTS, "count_waveform_scaler.pkl"))
    scaler_t0 = joblib.load(os.path.join(cfg.DIR_TRAINING_PLOTS, "t0_scaler.pkl"))
    scaler_amp = joblib.load(os.path.join(cfg.DIR_TRAINING_PLOTS, "amp_scaler.pkl"))

    X_sig = scaler_wave.transform(waveforms)[..., None]
    X_cnt = scaler_cwave.transform(waveforms)[..., None]
    counts = np.argmax(count_model.predict(X_cnt, verbose=0), axis=1)
    raw = signal_model.predict(X_sig, verbose=0)
    t0 = scaler_t0.inverse_transform(raw[:, 0::2])
    amp = scaler_amp.inverse_transform(raw[:, 1::2])

    return [
        [(float(t0[i, j]), float(amp[i, j])) for j in range(int(counts[i]))]
        for i in range(len(waveforms))
    ]


def _score_method(
    name: str,
    preds: list[list[tuple[float, float]]],
    truths: list[list[tuple[float, float]]],
) -> dict:
    all_t0, all_amp = [], []
    by_count: dict[int, list[float]] = {k: [] for k in range(7)}
    missed = spurious = 0
    count_right = count_total = 0
    total_true = total_pred = 0

    for truth, pred in zip(truths, preds):
        t0e, ampe, m, s = _match_and_score(truth, pred)
        all_t0.extend(t0e)
        all_amp.extend(ampe)
        missed += m
        spurious += s
        total_true += len(truth)
        total_pred += len(pred)
        count_total += 1
        if len(pred) == len(truth):
            count_right += 1
        by_count[len(truth)].extend(np.abs(t0e).tolist())

    def _stat(arr: list[float], fn) -> float:
        return float(fn(np.asarray(arr))) if arr else float("nan")

    return {
        "method": name,
        "count_acc": count_right / max(count_total, 1),
        "t0_mae_ns": _stat(all_t0, lambda a: np.mean(np.abs(a))),
        "t0_rmse_ns": _stat(all_t0, lambda a: np.sqrt(np.mean(a ** 2))),
        "t0_median_ns": _stat(all_t0, lambda a: np.median(np.abs(a))),
        "amp_mae": _stat(all_amp, lambda a: np.mean(np.abs(a))),
        "amp_median": _stat(all_amp, lambda a: np.median(np.abs(a))),
        "miss_rate": missed / max(total_true, 1),
        "fp_rate": spurious / max(total_pred, 1),
        "per_count_t0_mae": {k: float(np.mean(v)) if v else float("nan")
                             for k, v in by_count.items()},
    }


def main(
    subset: int | None = None,
    threshold: float = 2.0,
    output_dir: str | None = None,
):
    """Run the comparison. `subset` limits to N val waveforms (speed)."""
    signals_path = os.path.join(cfg.DIR_ML_DATA, "training_data_signals.npz")
    counts_path = os.path.join(cfg.DIR_ML_DATA, "training_data_counts.npz")
    data = np.load(signals_path)
    waveforms = data["waveforms"]
    labels = data["labels"]
    counts = np.load(counts_path)["labels"]

    idx = np.arange(len(waveforms))
    _, val_idx = train_test_split(idx, test_size=cfg.TEST_SIZE, random_state=cfg.RANDOM_STATE)
    if subset is not None:
        val_idx = val_idx[:subset]

    X_val = waveforms[val_idx]
    y_val = labels[val_idx]
    counts_val = counts[val_idx]

    truths: list[list[tuple[float, float]]] = []
    for i in range(len(val_idx)):
        k = int(counts_val[i])
        truths.append([(float(y_val[i, j, 0]), float(y_val[i, j, 1])) for j in range(k)])

    logger.info(f"Comparing on {len(X_val)} validation waveforms (threshold={threshold})")

    logger.info("Running matched-filter baseline...")
    mf_preds = [matched_filter(w, threshold) for w in X_val]

    logger.info("Running iterative-CLEAN baseline...")
    cl_preds = [iterative_clean(w, threshold=threshold) for w in X_val]

    logger.info("Running ML model...")
    ml_preds = _ml_predict(X_val)

    rows = [
        _score_method("Matched filter", mf_preds, truths),
        _score_method("Iterative CLEAN", cl_preds, truths),
        _score_method("ML (CNN + PIT)", ml_preds, truths),
    ]

    # Print summary table
    summary_cols = ["method", "count_acc", "t0_mae_ns", "t0_rmse_ns",
                    "amp_mae", "miss_rate", "fp_rate"]
    summary = pd.DataFrame([{c: r[c] for c in summary_cols} for r in rows])
    print("\n" + "=" * 82)
    print("BASELINE COMPARISON  (validation set, threshold = %.2f)" % threshold)
    print("=" * 82)
    print(summary.to_string(index=False, float_format=lambda v: f"{v:8.4f}"))

    print("\nPer-count t0 MAE (ns):")
    pc = pd.DataFrame({r["method"]: r["per_count_t0_mae"] for r in rows}).T
    pc.columns = [f"k={k}" for k in pc.columns]
    print(pc.to_string(float_format=lambda v: f"{v:7.2f}"))

    # Save artifacts
    out_dir = output_dir or cfg.DIR_COMPARISON_PLOTS
    os.makedirs(out_dir, exist_ok=True)

    summary.to_csv(os.path.join(out_dir, "baseline_comparison.csv"), index=False)
    pc.to_csv(os.path.join(out_dir, "baseline_per_count_t0_mae.csv"))

    # Markdown report (fallback to plain-text if `tabulate` isn't installed)
    def _md(df: pd.DataFrame, fmt: str, keep_index: bool = True) -> str:
        try:
            return df.to_markdown(index=keep_index, floatfmt=fmt)
        except ImportError:
            return "```\n" + df.to_string(float_format=lambda v: format(v, fmt)) + "\n```"

    report_path = os.path.join(out_dir, "baseline_comparison.md")
    with open(report_path, "w") as f:
        f.write("# Baseline Comparison\n\n")
        f.write(f"Validation waveforms: **{len(X_val)}**, threshold: **{threshold}**\n\n")
        f.write("## Summary\n\n")
        f.write(_md(summary, ".4f", keep_index=False) + "\n\n")
        f.write("## Per-count t0 MAE (ns)\n\n")
        f.write(_md(pc, ".2f") + "\n")
    logger.info(f"Saved report: {report_path}")

    # Plot per-count t0 MAE
    fig, ax = plt.subplots(figsize=(8, 5))
    for r in rows:
        ks = sorted(r["per_count_t0_mae"].keys())
        vals = [r["per_count_t0_mae"][k] for k in ks]
        ax.plot(ks, vals, marker="o", label=r["method"])
    ax.set_xlabel("Number of signals in waveform")
    ax.set_ylabel("t0 MAE (ns)")
    ax.set_title("Timing resolution vs. pulse multiplicity")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plot_path = os.path.join(out_dir, "baseline_per_count_t0_mae.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=120)
    plt.close()
    logger.info(f"Saved plot: {plot_path}")

    return {"summary": summary.to_dict(orient="records"),
            "per_count": pc.to_dict(orient="index")}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()

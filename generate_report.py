"""Generate a human-readable evaluation report for an experiment."""

import json
import os
import csv
import logging

logger = logging.getLogger(__name__)


def _load_json(path):
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


def _load_csv_last_row(path):
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return rows[-1] if rows else None


def _fmt(val, decimals=4):
    if val is None:
        return "N/A"
    try:
        return f"{float(val):.{decimals}f}"
    except (ValueError, TypeError):
        return str(val)


def _pct(val, decimals=1):
    if val is None:
        return "N/A"
    try:
        return f"{float(val) * 100:.{decimals}f}%"
    except (ValueError, TypeError):
        return str(val)


def generate_report(experiment_dir):
    """Generate a markdown evaluation report for the given experiment directory.

    Returns the report as a string and saves it to experiment_dir/report.md.
    """
    config = _load_json(os.path.join(experiment_dir, "config.json"))
    metrics = _load_json(os.path.join(experiment_dir, "metrics.json"))
    error_report = _load_json(os.path.join(experiment_dir, "error_analysis", "error_analysis_report.json"))

    count_history = _load_csv_last_row(os.path.join(experiment_dir, "training_plots", "signal_count_model_history.csv"))
    signal_history = _load_csv_last_row(os.path.join(experiment_dir, "training_plots", "signal_model_history.csv"))

    lines = []
    w = lines.append

    exp_name = os.path.basename(experiment_dir)
    w(f"# Evaluation Report: {exp_name}")
    w("")

    # --- Configuration ---
    w("## Configuration")
    w("")
    if config:
        key_params = [
            ("num_waveforms", "Waveforms"),
            ("epochs", "Epochs"),
            ("batch_size", "Batch size"),
            ("test_size", "Test split"),
            ("min_spacing", "Min spacing (ns)"),
            ("max_signals", "Max signals"),
            ("noise_std", "Noise std"),
            ("varied_noise", "Varied noise"),
        ]
        w("| Parameter | Value |")
        w("|-----------|-------|")
        for key, label in key_params:
            if key in config:
                w(f"| {label} | {config[key]} |")
        pit_status = config.get("use_pit")
        if pit_status is not None:
            w(f"| PIT (Hungarian) | {pit_status} |")
        w("")
    else:
        w("*Config not found.*\n")

    # --- Timing ---
    w("## Timing")
    w("")
    if metrics and "timings" in metrics:
        w("| Step | Time (s) |")
        w("|------|----------|")
        for step, elapsed in metrics["timings"].items():
            w(f"| {step} | {elapsed} |")
        if "total_time_s" in metrics:
            w(f"| **Total** | **{metrics['total_time_s']}** |")
        w("")
    else:
        w("*Timing data not found.*\n")

    # --- Count Model ---
    w("## Count Model")
    w("")

    if metrics and "training_count_model" in metrics:
        cm = metrics["training_count_model"]
        w(f"- **Accuracy**: {_pct(cm.get('accuracy'))}")
        if "roc_auc" in cm:
            w(f"- **ROC AUC**: {_fmt(cm['roc_auc'])}")
        w("")

    if count_history:
        w(f"- Final train accuracy: {_pct(count_history.get('accuracy'))}")
        w(f"- Final val accuracy: {_pct(count_history.get('val_accuracy'))}")
        w(f"- Final train loss: {_fmt(count_history.get('loss'))}")
        w(f"- Final val loss: {_fmt(count_history.get('val_loss'))}")
        w(f"- Final learning rate: {count_history.get('learning_rate', 'N/A')}")
        w("")

    if error_report and "count_model" in error_report:
        cm = error_report["count_model"]
        if "per_class_accuracy" in cm:
            w("### Per-Class Accuracy")
            w("")
            w("| Count | Accuracy |")
            w("|-------|----------|")
            for cls, acc in sorted(cm["per_class_accuracy"].items(), key=lambda x: int(x[0])):
                w(f"| {cls} | {_pct(acc)} |")
            w("")

        if "top_misclassifications" in cm:
            w("### Top Misclassifications")
            w("")
            w("| True | Predicted | Count |")
            w("|------|-----------|-------|")
            for m in cm["top_misclassifications"][:5]:
                w(f"| {m['true']} | {m['pred']} | {m['count']} |")
            w("")

    # --- Signal Model ---
    w("## Signal Model")
    w("")

    if metrics and "training_signal_model" in metrics:
        sm = metrics["training_signal_model"]
        w("### Overall Metrics")
        w("")
        w("| Metric | t0 (ns) | Amplitude |")
        w("|--------|---------|-----------|")
        w(f"| MAE | {_fmt(sm.get('t0_mae'))} | {_fmt(sm.get('amp_mae'))} |")
        w(f"| RMSE | {_fmt(sm.get('t0_rmse'))} | {_fmt(sm.get('amp_rmse'))} |")
        w(f"| Pearson r | {_fmt(sm.get('t0_pearson'))} | {_fmt(sm.get('amp_pearson'))} |")
        w(f"| Spearman r | {_fmt(sm.get('t0_spearman'))} | {_fmt(sm.get('amp_spearman'))} |")
        w("")

    if signal_history:
        w(f"- Final train loss: {_fmt(signal_history.get('loss'))}")
        w(f"- Final val loss: {_fmt(signal_history.get('val_loss'))}")
        w(f"- Final train MAE: {_fmt(signal_history.get('mae'))}")
        w(f"- Final val MAE: {_fmt(signal_history.get('val_mae'))}")
        w("")

    # --- Comparison metrics (Hungarian-matched) ---
    if metrics and "comparing_predictions" in metrics:
        cp = metrics["comparing_predictions"]
        w("### Comparison Metrics (Hungarian-matched)")
        w("")
        w("| Metric | t0 (ns) | Amplitude |")
        w("|--------|---------|-----------|")
        w(f"| MAE | {_fmt(cp.get('t0_mae'))} | {_fmt(cp.get('amp_mae'))} |")
        w(f"| RMSE | {_fmt(cp.get('t0_rmse'))} | {_fmt(cp.get('amp_rmse'))} |")
        w(f"| R² | {_fmt(cp.get('t0_r2'))} | {_fmt(cp.get('amp_r2'))} |")
        w(f"| Pearson r | {_fmt(cp.get('t0_pearson'))} | {_fmt(cp.get('amp_pearson'))} |")
        if "count_accuracy" in cp:
            w(f"\n- **Count accuracy**: {_pct(cp['count_accuracy'])}")
        if "n_signal_pairs" in cp:
            w(f"- **Matched signal pairs**: {cp['n_signal_pairs']}")
        w("")

    # --- Error Analysis Details ---
    if error_report:
        w("## Error Analysis")
        w("")

        # Per-slot
        if "per_slot" in error_report:
            ps = error_report["per_slot"]
            w("### Per-Slot MAE")
            w("")
            w("| Slot | t0 MAE (ns) | Amp MAE | t0 RMSE (ns) | Amp RMSE |")
            w("|------|-------------|---------|---------------|----------|")
            n_slots = len(ps.get("slot_t0_mae", []))
            for i in range(n_slots):
                t0m = ps["slot_t0_mae"][i] if i < len(ps.get("slot_t0_mae", [])) else None
                am = ps["slot_amp_mae"][i] if i < len(ps.get("slot_amp_mae", [])) else None
                t0r = ps["slot_t0_rmse"][i] if i < len(ps.get("slot_t0_rmse", [])) else None
                ar = ps["slot_amp_rmse"][i] if i < len(ps.get("slot_amp_rmse", [])) else None
                w(f"| {i} | {_fmt(t0m)} | {_fmt(am)} | {_fmt(t0r)} | {_fmt(ar)} |")
            w("")

        # Error vs count
        if "error_vs_count" in error_report:
            ec = error_report["error_vs_count"]
            w("### Error by Signal Count")
            w("")
            w("| Count | t0 Bias (ns) | t0 Std (ns) | Amp Bias | Amp Std |")
            w("|-------|-------------|-------------|----------|---------|")
            counts = sorted(ec.get("per_count_t0_bias", {}).keys(), key=int)
            for c in counts:
                t0b = ec["per_count_t0_bias"].get(c)
                t0s = ec["per_count_t0_std"].get(c)
                ab = ec["per_count_amp_bias"].get(c)
                as_ = ec["per_count_amp_std"].get(c)
                w(f"| {c} | {_fmt(t0b)} | {_fmt(t0s)} | {_fmt(ab)} | {_fmt(as_)} |")
            w("")

        # Temporal profile
        if "temporal_profile" in error_report:
            tp = error_report["temporal_profile"]
            if "binned_t0_mae_by_position" in tp:
                w("### Temporal Error Profile")
                w("")
                w("| Time Bin | t0 MAE (ns) |")
                w("|----------|-------------|")
                for bin_name, mae in tp["binned_t0_mae_by_position"].items():
                    w(f"| {bin_name} | {_fmt(mae)} |")
                w("")

        # Spacing
        if "spacing" in error_report:
            sp = error_report["spacing"]
            w("### Signal Spacing Impact")
            w("")
            w(f"- Spacing vs t0 MAE (Spearman): {_fmt(sp.get('spacing_t0_mae_spearman'))}")
            w(f"- Spacing vs amp MAE (Spearman): {_fmt(sp.get('spacing_amp_mae_spearman'))}")
            w(f"- Multi-signal waveforms: {sp.get('n_multi_signal_waveforms', 'N/A')}")
            w("")

        # Worst cases
        if "worst_cases" in error_report:
            wc = error_report["worst_cases"]
            if "error_percentiles" in wc:
                w("### Error Percentiles")
                w("")
                w("| Percentile | Total Error |")
                w("|------------|-------------|")
                for p, v in wc["error_percentiles"].items():
                    w(f"| {p} | {_fmt(v, 2)} |")
                w("")

        # Residuals
        if "residuals" in error_report:
            res = error_report["residuals"]
            w("### Residual Statistics")
            w("")
            w("| Metric | t0 | Amplitude |")
            w("|--------|----|-----------| ")
            t0r = res.get("t0_residual", {})
            ar = res.get("amp_residual", {})
            w(f"| Mean | {_fmt(t0r.get('mean'))} | {_fmt(ar.get('mean'))} |")
            w(f"| Std | {_fmt(t0r.get('std'))} | {_fmt(ar.get('std'))} |")
            w(f"| Skewness | {_fmt(t0r.get('skewness'))} | {_fmt(ar.get('skewness'))} |")
            w(f"| Kurtosis | {_fmt(t0r.get('kurtosis'))} | {_fmt(ar.get('kurtosis'))} |")
            w("")

    # --- Summary ---
    w("---")
    w(f"*Report generated for experiment: {exp_name}*")

    report = "\n".join(lines)

    report_path = os.path.join(experiment_dir, "report.md")
    with open(report_path, "w") as f:
        f.write(report)
    logger.info(f"Evaluation report saved to '{report_path}'")

    return report


def main(experiment_dir):
    return generate_report(experiment_dir)


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) < 2:
        print("Usage: python generate_report.py <experiment_dir>")
        sys.exit(1)
    report = main(sys.argv[1])
    print(report)

#!/usr/bin/env python3
"""CLI for the ADC waveform signal extraction ML pipeline."""

import argparse
import json
import logging
import os
import shutil
import sys
import time
from datetime import datetime
import config as cfg

logger = logging.getLogger(__name__)


def setup_logging(log_file="pipeline.log"):
    """Configure logging to both console and file."""
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    root.addHandler(console)

    os.makedirs(cfg.DIR_LOGS, exist_ok=True)
    file_handler = logging.FileHandler(os.path.join(cfg.DIR_LOGS, log_file))
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)


def _timed(label, func, *a, **kw):
    """Run func and log elapsed time."""
    logger.debug(f"Starting: {label}")
    t0 = time.time()
    result = func(*a, **kw)
    elapsed = time.time() - t0
    logger.info(f"Finished: {label} ({elapsed:.1f}s)")
    return result


def cmd_generate(args):
    from gen_wave import generate_dataset
    _timed("generate waveforms", generate_dataset,
        num_waveforms=args.num_waveforms,
        output_dir=args.output_dir,
        noise_std=args.noise_std,
        baseline=args.baseline,
        min_spacing=args.min_spacing,
        max_signals=args.max_signals,
        varied_noise=args.varied_noise,
    )


def cmd_baseline(args):
    from baseline_subtract import subtract_baseline
    input_dir = getattr(args, 'baseline_input_dir', None) or args.input_dir
    output_dir = getattr(args, 'baseline_output_dir', None) or args.output_dir
    _timed("baseline subtraction", subtract_baseline,
        input_dir=input_dir,
        output_dir=output_dir,
        window_size=args.window_size,
        quantile=args.quantile,
    )


def cmd_prepare(args):
    from prepare_ml_dataset import main
    input_dir = getattr(args, 'prepare_input_dir', None) or args.input_dir
    output_dir = getattr(args, 'prepare_output_dir', None) or args.output_dir
    _timed("prepare dataset", main,
        input_dir=input_dir,
        truth_dir=args.truth_dir,
        output_dir=output_dir,
        max_signals=args.max_signals,
    )


def cmd_train_count(args):
    from train_count_model import main
    log_dir = os.path.join(cfg.DIR_TENSORBOARD, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), "count_model")
    os.makedirs(log_dir, exist_ok=True)
    use_gpu = getattr(args, 'use_gpu', None)
    return _timed("train count model", main,
        epochs=args.epochs, batch_size=args.batch_size, test_size=args.test_size, log_dir=log_dir, use_gpu=use_gpu)


def cmd_train_signal(args):
    from train_signal_model import main
    log_dir = os.path.join(cfg.DIR_TENSORBOARD, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), "signal_model")
    os.makedirs(log_dir, exist_ok=True)
    use_pit = getattr(args, 'use_pit', None)
    use_gpu = getattr(args, 'use_gpu', None)
    return _timed("train signal model", main,
        epochs=args.epochs, batch_size=args.batch_size, test_size=args.test_size, log_dir=log_dir, use_pit=use_pit, use_gpu=use_gpu)


def cmd_compare(args):
    from compare_signal_predictions import main
    return _timed("compare predictions", main)


def cmd_plot(args):
    from plot_individual_waveform import main
    _timed("plot individual waveforms", main,
        start=args.start, end=args.end)


def cmd_analyze(args):
    from error_analysis.analyze import main
    return _timed("error analysis", main, experiment_dir=args.experiment)


def cmd_baselines(args):
    from compare_baselines import main
    return _timed("compare baselines", main,
                  subset=args.subset, threshold=args.threshold,
                  output_dir=args.output_dir)


def cmd_report(args):
    from generate_report import generate_report
    report = generate_report(args.experiment)
    print(report)


def _save_experiment(args, all_metrics, timings, total_time):
    """Collect all results into a timestamped experiment folder."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    name = getattr(args, 'experiment_name', None)
    folder_name = f"{timestamp}_{name}" if name else timestamp
    exp_dir = os.path.join(cfg.DIR_EXPERIMENTS, folder_name)
    os.makedirs(exp_dir, exist_ok=True)

    # Save config
    config = {k: v for k, v in vars(args).items()
              if k not in ('func', 'command', 'baseline_input_dir', 'baseline_output_dir',
                           'prepare_input_dir', 'prepare_output_dir', 'experiment_name')}
    with open(os.path.join(exp_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Save metrics
    all_metrics['timings'] = {msg: round(elapsed, 1) for msg, elapsed in timings}
    all_metrics['total_time_s'] = round(total_time, 1)
    with open(os.path.join(exp_dir, "metrics.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)

    # Copy plots
    for src_dir in (cfg.DIR_TRAINING_PLOTS, cfg.DIR_COMPARISON_PLOTS, cfg.DIR_WAVEFORM_INSPECTION):
        if os.path.isdir(src_dir):
            dst = os.path.join(exp_dir, src_dir)
            shutil.copytree(src_dir, dst, dirs_exist_ok=True)

    # Copy TensorBoard logs
    tb_run_dir = os.path.join(cfg.DIR_TENSORBOARD, name if name else folder_name)
    if os.path.isdir(tb_run_dir):
        shutil.copytree(tb_run_dir, os.path.join(exp_dir, "tensorboard"), dirs_exist_ok=True)

    # Copy log
    log_path = os.path.join(cfg.DIR_LOGS, "pipeline.log")
    if os.path.isfile(log_path):
        shutil.copy2(log_path, os.path.join(exp_dir, "pipeline.log"))

    logger.info(f"Experiment saved to '{exp_dir}/'")
    return exp_dir


def _train_count_worker(epochs, batch_size, test_size, log_dir=None, use_gpu=None):
    """Worker for parallel training — runs in a subprocess."""
    from train_count_model import main
    return main(epochs=epochs, batch_size=batch_size, test_size=test_size, log_dir=log_dir, use_gpu=use_gpu)


def _train_signal_worker(epochs, batch_size, test_size, log_dir=None, use_pit=None, use_gpu=None):
    """Worker for parallel training — runs in a subprocess."""
    from train_signal_model import main
    return main(epochs=epochs, batch_size=batch_size, test_size=test_size, log_dir=log_dir, use_pit=use_pit, use_gpu=use_gpu)


def cmd_run_all(args):
    from concurrent.futures import ProcessPoolExecutor

    # Set correct per-step directories so shared args don't conflict
    args.baseline_input_dir = args.output_dir              # waveform_raw
    args.baseline_output_dir = cfg.DIR_BASELINE_REMOVED
    args.prepare_input_dir = cfg.DIR_BASELINE_REMOVED
    args.prepare_output_dir = cfg.DIR_ML_DATA

    total_t0 = time.time()
    timings = []
    all_metrics = {}

    # Steps 1-3: sequential (each depends on the previous)
    sequential_steps = [
        ("Step 1/7: Generating waveforms", cmd_generate),
        ("Step 2/7: Subtracting baselines", cmd_baseline),
        ("Step 3/7: Preparing ML dataset", cmd_prepare),
    ]
    for msg, func in sequential_steps:
        logger.info("=" * 50)
        logger.info(msg)
        logger.info("=" * 50)
        t0 = time.time()
        func(args)
        timings.append((msg, time.time() - t0))

    # Steps 4 & 5: parallel (independent models)
    logger.info("=" * 50)
    logger.info("Steps 4-5/7: Training both models in parallel")
    logger.info("=" * 50)

    # TensorBoard log dirs
    tb_run_name = getattr(args, 'experiment_name', None) or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    tb_count_dir = os.path.join(cfg.DIR_TENSORBOARD, tb_run_name, "count_model")
    tb_signal_dir = os.path.join(cfg.DIR_TENSORBOARD, tb_run_name, "signal_model")
    os.makedirs(tb_count_dir, exist_ok=True)
    os.makedirs(tb_signal_dir, exist_ok=True)

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=2) as executor:
        use_gpu = getattr(args, 'use_gpu', None)
        future_count = executor.submit(
            _train_count_worker, args.epochs, args.batch_size, args.test_size, tb_count_dir, use_gpu)
        use_pit = getattr(args, 'use_pit', None)
        future_signal = executor.submit(
            _train_signal_worker, args.epochs, args.batch_size, args.test_size, tb_signal_dir, use_pit, use_gpu)
        count_metrics = future_count.result()
        signal_metrics = future_signal.result()
    parallel_elapsed = time.time() - t0
    timings.append(("Steps 4-5/7: Training both models (parallel)", parallel_elapsed))
    if isinstance(count_metrics, dict):
        all_metrics["training_count_model"] = count_metrics
    if isinstance(signal_metrics, dict):
        all_metrics["training_signal_model"] = signal_metrics

    # Steps 6-7: sequential (need both models)
    final_steps = [
        ("Step 6/7: Comparing predictions", cmd_compare),
        ("Step 7/7: Plotting individual waveforms", cmd_plot),
    ]
    for msg, func in final_steps:
        logger.info("=" * 50)
        logger.info(msg)
        logger.info("=" * 50)
        t0 = time.time()
        result = func(args)
        timings.append((msg, time.time() - t0))
        if isinstance(result, dict):
            step_key = msg.split(": ", 1)[1].lower().replace(" ", "_")
            all_metrics[step_key] = result

    total = time.time() - total_t0
    logger.info("=" * 50)
    logger.info("Pipeline timing summary:")
    for msg, elapsed in timings:
        logger.info(f"  {msg}: {elapsed:.1f}s")
    logger.info(f"  Total: {total:.1f}s")
    logger.info("=" * 50)

    exp_dir = _save_experiment(args, all_metrics, timings, total)

    # Run error analysis on the saved experiment
    logger.info("=" * 50)
    logger.info("Step 8: Running error analysis")
    logger.info("=" * 50)
    from error_analysis.analyze import main as analyze_main
    _timed("error analysis", analyze_main, experiment_dir=exp_dir)

    # Generate evaluation report
    logger.info("=" * 50)
    logger.info("Step 9: Generating evaluation report")
    logger.info("=" * 50)
    from generate_report import generate_report
    report = _timed("evaluation report", generate_report, experiment_dir=exp_dir)
    logger.info(f"\n{report}")


def build_parser():
    parser = argparse.ArgumentParser(
        prog="cli.py",
        description="ADC Waveform Signal Extraction ML Pipeline",
    )
    subparsers = parser.add_subparsers(dest="command", help="Pipeline step to run")

    # generate
    p = subparsers.add_parser("generate", help="Generate synthetic waveforms")
    p.add_argument("--num-waveforms", type=int, default=cfg.NUM_WAVEFORMS)
    p.add_argument("--output-dir", default=cfg.DIR_RAW)
    p.add_argument("--noise-std", type=float, default=cfg.NOISE_STD)
    p.add_argument("--baseline", type=float, default=cfg.BASELINE)
    p.add_argument("--min-spacing", type=float, default=cfg.MIN_SPACING)
    p.add_argument("--max-signals", type=int, default=cfg.MAX_SIGNALS)
    p.add_argument("--varied-noise", action="store_true", default=cfg.VARIED_NOISE)
    p.add_argument("--no-varied-noise", dest="varied_noise", action="store_false")
    p.set_defaults(func=cmd_generate)

    # baseline
    p = subparsers.add_parser("baseline", help="Subtract baselines from waveforms")
    p.add_argument("--input-dir", default=cfg.DIR_RAW)
    p.add_argument("--output-dir", default=cfg.DIR_BASELINE_REMOVED)
    p.add_argument("--window-size", type=int, default=cfg.WINDOW_SIZE)
    p.add_argument("--quantile", type=float, default=cfg.QUANTILE)
    p.set_defaults(func=cmd_baseline)

    # prepare
    p = subparsers.add_parser("prepare", help="Create .npz training datasets")
    p.add_argument("--input-dir", default=cfg.DIR_BASELINE_REMOVED)
    p.add_argument("--truth-dir", default=cfg.DIR_RAW)
    p.add_argument("--output-dir", default=cfg.DIR_ML_DATA)
    p.add_argument("--max-signals", type=int, default=cfg.MAX_SIGNALS)
    p.set_defaults(func=cmd_prepare)

    # train-count
    p = subparsers.add_parser("train-count", help="Train signal count classifier")
    p.add_argument("--epochs", type=int, default=cfg.COUNT_MODEL_EPOCHS)
    p.add_argument("--batch-size", type=int, default=cfg.COUNT_MODEL_BATCH_SIZE)
    p.add_argument("--test-size", type=float, default=cfg.TEST_SIZE)
    p.add_argument("--gpu", dest="use_gpu", action="store_true", default=None,
                   help="Use GPU for training (auto-detect if available)")
    p.add_argument("--no-gpu", dest="use_gpu", action="store_false",
                   help="Force CPU-only training")
    p.set_defaults(func=cmd_train_count)

    # train-signal
    p = subparsers.add_parser("train-signal", help="Train signal parameter regressor")
    p.add_argument("--epochs", type=int, default=cfg.SIGNAL_MODEL_EPOCHS)
    p.add_argument("--batch-size", type=int, default=cfg.SIGNAL_MODEL_BATCH_SIZE)
    p.add_argument("--test-size", type=float, default=cfg.TEST_SIZE)
    p.add_argument("--pit", dest="use_pit", action="store_true", default=None,
                   help="Enable permutation-invariant training (Hungarian matching)")
    p.add_argument("--no-pit", dest="use_pit", action="store_false",
                   help="Disable PIT, use standard model.fit()")
    p.add_argument("--gpu", dest="use_gpu", action="store_true", default=None,
                   help="Use GPU for training (auto-detect if available)")
    p.add_argument("--no-gpu", dest="use_gpu", action="store_false",
                   help="Force CPU-only training")
    p.set_defaults(func=cmd_train_signal)

    # compare
    p = subparsers.add_parser("compare", help="Generate prediction comparison plots")
    p.set_defaults(func=cmd_compare)

    # plot
    p = subparsers.add_parser("plot", help="Plot individual waveforms with predictions")
    p.add_argument("--start", type=int, default=cfg.PLOT_START)
    p.add_argument("--end", type=int, default=cfg.PLOT_END)
    p.set_defaults(func=cmd_plot)

    # analyze
    p = subparsers.add_parser("analyze", help="Run error analysis on an experiment folder")
    p.add_argument("--experiment", required=True, help="Path to experiment folder")
    p.set_defaults(func=cmd_analyze)

    # baselines
    p = subparsers.add_parser(
        "baselines",
        help="Compare the trained ML model against classical baselines"
    )
    p.add_argument("--subset", type=int, default=None,
                   help="Use only the first N validation waveforms (faster)")
    p.add_argument("--threshold", type=float, default=2.0,
                   help="Amplitude threshold for peak detection (in pulse-scale units)")
    p.add_argument("--output-dir", default=cfg.DIR_COMPARISON_PLOTS,
                   help="Where to save the comparison CSV/markdown/plot")
    p.set_defaults(func=cmd_baselines)

    # report
    p = subparsers.add_parser("report", help="Generate evaluation report for an experiment")
    p.add_argument("--experiment", required=True, help="Path to experiment folder")
    p.set_defaults(func=cmd_report)

    # run-all
    p = subparsers.add_parser("run-all", help="Run the full pipeline end-to-end")
    p.add_argument("--num-waveforms", type=int, default=cfg.NUM_WAVEFORMS)
    p.add_argument("--output-dir", default=cfg.DIR_RAW)
    p.add_argument("--noise-std", type=float, default=cfg.NOISE_STD)
    p.add_argument("--baseline", type=float, default=cfg.BASELINE)
    p.add_argument("--min-spacing", type=float, default=cfg.MIN_SPACING)
    p.add_argument("--input-dir", default=cfg.DIR_BASELINE_REMOVED)
    p.add_argument("--truth-dir", default=cfg.DIR_RAW)
    p.add_argument("--window-size", type=int, default=cfg.WINDOW_SIZE)
    p.add_argument("--quantile", type=float, default=cfg.QUANTILE)
    p.add_argument("--max-signals", type=int, default=cfg.MAX_SIGNALS)
    p.add_argument("--varied-noise", action="store_true", default=cfg.VARIED_NOISE)
    p.add_argument("--no-varied-noise", dest="varied_noise", action="store_false")
    p.add_argument("--epochs", type=int, default=cfg.COUNT_MODEL_EPOCHS)
    p.add_argument("--batch-size", type=int, default=cfg.COUNT_MODEL_BATCH_SIZE)
    p.add_argument("--test-size", type=float, default=cfg.TEST_SIZE)
    p.add_argument("--start", type=int, default=cfg.PLOT_START)
    p.add_argument("--end", type=int, default=cfg.PLOT_END)
    p.add_argument("--experiment-name", default=None, help="Optional name for the experiment folder")
    p.add_argument("--pit", dest="use_pit", action="store_true", default=None,
                   help="Enable permutation-invariant training (Hungarian matching)")
    p.add_argument("--no-pit", dest="use_pit", action="store_false",
                   help="Disable PIT, use standard model.fit()")
    p.add_argument("--gpu", dest="use_gpu", action="store_true", default=None,
                   help="Use GPU for training (auto-detect if available)")
    p.add_argument("--no-gpu", dest="use_gpu", action="store_false",
                   help="Force CPU-only training")
    p.set_defaults(func=cmd_run_all)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    setup_logging()
    try:
        args.func(args)
    except Exception:
        logger.exception("Pipeline failed")
        sys.exit(1)


if __name__ == "__main__":
    main()

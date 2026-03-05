#!/usr/bin/env python3
"""CLI for the ADC waveform signal extraction ML pipeline."""

import argparse
import logging
import os
import sys
import time

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

    os.makedirs("logs", exist_ok=True)
    file_handler = logging.FileHandler(os.path.join("logs", log_file))
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
    from genWave import generate_dataset
    _timed("generate waveforms", generate_dataset,
        num_waveforms=args.num_waveforms,
        output_dir=args.output_dir,
        noise_std=args.noise_std,
        baseline=args.baseline,
        min_spacing=args.min_spacing,
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
    _timed("train count model", main,
        epochs=args.epochs, batch_size=args.batch_size, test_size=args.test_size)


def cmd_train_signal(args):
    from train_signal_model import main
    _timed("train signal model", main,
        epochs=args.epochs, batch_size=args.batch_size, test_size=args.test_size)


def cmd_compare(args):
    from compare_signal_predictions import main
    _timed("compare predictions", main)


def cmd_plot(args):
    from plot_individual_waveform import main
    _timed("plot individual waveforms", main,
        start=args.start, end=args.end)


def cmd_run_all(args):
    # Set correct per-step directories so shared args don't conflict
    args.baseline_input_dir = args.output_dir              # waveform_raw
    args.baseline_output_dir = "waveform_baseline_removed"
    args.prepare_input_dir = "waveform_baseline_removed"
    args.prepare_output_dir = "ml_training_data"

    total_t0 = time.time()
    steps = [
        ("Step 1/7: Generating waveforms", cmd_generate),
        ("Step 2/7: Subtracting baselines", cmd_baseline),
        ("Step 3/7: Preparing ML dataset", cmd_prepare),
        ("Step 4/7: Training count model", cmd_train_count),
        ("Step 5/7: Training signal model", cmd_train_signal),
        ("Step 6/7: Comparing predictions", cmd_compare),
        ("Step 7/7: Plotting individual waveforms", cmd_plot),
    ]
    timings = []
    for msg, func in steps:
        logger.info("=" * 50)
        logger.info(msg)
        logger.info("=" * 50)
        t0 = time.time()
        func(args)
        timings.append((msg, time.time() - t0))

    total = time.time() - total_t0
    logger.info("=" * 50)
    logger.info("Pipeline timing summary:")
    for msg, elapsed in timings:
        logger.info(f"  {msg}: {elapsed:.1f}s")
    logger.info(f"  Total: {total:.1f}s")
    logger.info("=" * 50)


def build_parser():
    parser = argparse.ArgumentParser(
        prog="cli.py",
        description="ADC Waveform Signal Extraction ML Pipeline",
    )
    subparsers = parser.add_subparsers(dest="command", help="Pipeline step to run")

    # generate
    p = subparsers.add_parser("generate", help="Generate synthetic waveforms")
    p.add_argument("--num-waveforms", type=int, default=50000)
    p.add_argument("--output-dir", default="waveform_raw")
    p.add_argument("--noise-std", type=float, default=0.5)
    p.add_argument("--baseline", type=float, default=200.0)
    p.add_argument("--min-spacing", type=float, default=0.0001)
    p.set_defaults(func=cmd_generate)

    # baseline
    p = subparsers.add_parser("baseline", help="Subtract baselines from waveforms")
    p.add_argument("--input-dir", default="waveform_raw")
    p.add_argument("--output-dir", default="waveform_baseline_removed")
    p.add_argument("--window-size", type=int, default=31)
    p.add_argument("--quantile", type=float, default=0.1)
    p.set_defaults(func=cmd_baseline)

    # prepare
    p = subparsers.add_parser("prepare", help="Create .npz training datasets")
    p.add_argument("--input-dir", default="waveform_baseline_removed")
    p.add_argument("--truth-dir", default="waveform_raw")
    p.add_argument("--output-dir", default="ml_training_data")
    p.add_argument("--max-signals", type=int, default=7)
    p.set_defaults(func=cmd_prepare)

    # train-count
    p = subparsers.add_parser("train-count", help="Train signal count classifier")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--test-size", type=float, default=0.2)
    p.set_defaults(func=cmd_train_count)

    # train-signal
    p = subparsers.add_parser("train-signal", help="Train signal parameter regressor")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--test-size", type=float, default=0.2)
    p.set_defaults(func=cmd_train_signal)

    # compare
    p = subparsers.add_parser("compare", help="Generate prediction comparison plots")
    p.set_defaults(func=cmd_compare)

    # plot
    p = subparsers.add_parser("plot", help="Plot individual waveforms with predictions")
    p.add_argument("--start", type=int, default=1)
    p.add_argument("--end", type=int, default=300)
    p.set_defaults(func=cmd_plot)

    # run-all
    p = subparsers.add_parser("run-all", help="Run the full pipeline end-to-end")
    p.add_argument("--num-waveforms", type=int, default=50000)
    p.add_argument("--output-dir", default="waveform_raw")
    p.add_argument("--noise-std", type=float, default=0.5)
    p.add_argument("--baseline", type=float, default=200.0)
    p.add_argument("--min-spacing", type=float, default=0.0001)
    p.add_argument("--input-dir", default="waveform_baseline_removed")
    p.add_argument("--truth-dir", default="waveform_raw")
    p.add_argument("--window-size", type=int, default=31)
    p.add_argument("--quantile", type=float, default=0.1)
    p.add_argument("--max-signals", type=int, default=7)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--start", type=int, default=1)
    p.add_argument("--end", type=int, default=300)
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

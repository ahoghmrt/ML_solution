import numpy as np
import os
import logging
import config as cfg

logger = logging.getLogger(__name__)

def main(input_dir=cfg.DIR_BASELINE_REMOVED, truth_dir=cfg.DIR_RAW, output_dir=cfg.DIR_ML_DATA, max_signals=cfg.MAX_SIGNALS):
    os.makedirs(output_dir, exist_ok=True)

    # Load baseline-subtracted waveforms
    bl_data = np.load(os.path.join(input_dir, "data.npz"))
    waveforms = bl_data["waveforms"]
    time = bl_data["time"]

    # Load truth from raw generation
    raw_data = np.load(os.path.join(truth_dir, "data.npz"))
    labels_regression = raw_data["truth"]   # (N, max_signals, 2)
    labels_count = raw_data["counts"]       # (N,)

    # Save separate datasets
    np.savez(os.path.join(output_dir, "training_data_signals.npz"),
             waveforms=waveforms,
             labels=labels_regression,
             time=time)

    np.savez(os.path.join(output_dir, "training_data_counts.npz"),
             waveforms=waveforms,
             labels=labels_count,
             time=time)

    logger.info(f"Saved training datasets to '{output_dir}/' ({waveforms.shape[0]} samples)")
    logger.debug(f"Signals - waveforms: {waveforms.shape}, labels: {labels_regression.shape}, time: {time.shape}")
    logger.debug(f"Counts - waveforms: {waveforms.shape}, labels: {labels_count.shape}, time: {time.shape}")


if __name__ == "__main__":
    main()

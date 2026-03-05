import numpy as np
import os
import pandas as pd
import logging
import config as cfg

logger = logging.getLogger(__name__)

# -------------------------------
# Rolling Quantile Baseline Function
# -------------------------------
def rolling_quantile_baseline(waveform, window_size=31, quantile=0.1):
    if window_size % 2 == 0:
        window_size += 1
    baseline = pd.Series(waveform).rolling(window=window_size, center=True, min_periods=1).quantile(quantile)
    return baseline.values

# -------------------------------
# Baseline Subtraction
# -------------------------------
def subtract_baseline(input_dir=cfg.DIR_RAW, output_dir=cfg.DIR_BASELINE_REMOVED, window_size=cfg.WINDOW_SIZE, quantile=cfg.QUANTILE):
    os.makedirs(output_dir, exist_ok=True)

    data = np.load(os.path.join(input_dir, "data.npz"))
    waveforms = data["waveforms"]
    time = data["time"]

    n = len(waveforms)
    logger.info(f"Processing {n} waveforms (window={window_size}, quantile={quantile})")

    subtracted = np.empty_like(waveforms)
    for i in range(n):
        baseline = rolling_quantile_baseline(waveforms[i], window_size, quantile)
        subtracted[i] = waveforms[i] - baseline

        if (i + 1) % 100 == 0 or (i + 1) == n:
            logger.debug(f"Processed {i + 1}/{n} waveforms")

    np.savez(os.path.join(output_dir, "data.npz"),
             waveforms=subtracted,
             time=time)

    logger.info(f"Baseline subtracted for {n} waveforms into '{output_dir}/'")

# -------------------------------
# RUN SCRIPT
# -------------------------------
if __name__ == "__main__":
    subtract_baseline()

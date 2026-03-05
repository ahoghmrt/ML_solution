import numpy as np
import os
import pandas as pd
import logging
import config as cfg

logger = logging.getLogger(__name__)


def subtract_baseline(input_dir=cfg.DIR_RAW, output_dir=cfg.DIR_BASELINE_REMOVED, window_size=cfg.WINDOW_SIZE, quantile=cfg.QUANTILE):
    os.makedirs(output_dir, exist_ok=True)

    data = np.load(os.path.join(input_dir, "data.npz"))
    waveforms = data["waveforms"]
    time = data["time"]

    if window_size % 2 == 0:
        window_size += 1

    n = len(waveforms)
    logger.info(f"Processing {n} waveforms (window={window_size}, quantile={quantile})")

    # Bulk rolling quantile: treat each waveform as a column, roll along rows (time axis)
    df = pd.DataFrame(waveforms.T)  # shape (120, N)
    baselines = df.rolling(window=window_size, center=True, min_periods=1).quantile(quantile).values.T
    subtracted = waveforms - baselines

    np.savez(os.path.join(output_dir, "data.npz"),
             waveforms=subtracted,
             time=time)

    logger.info(f"Baseline subtracted for {n} waveforms into '{output_dir}/'")


if __name__ == "__main__":
    subtract_baseline()

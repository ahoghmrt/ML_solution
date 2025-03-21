import numpy as np
import os
import pandas as pd
from glob import glob

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
def subtract_baseline(input_dir="waveform_raw", output_dir="waveform_baseline_removed", window_size=31, quantile=0.1):
    os.makedirs(output_dir, exist_ok=True)
    waveform_files = sorted(glob(os.path.join(input_dir, "waveform_*.txt")))

    for wf_file in waveform_files:
        # Load waveform
        data = np.loadtxt(wf_file, skiprows=1)
        time = data[:, 0]
        waveform = data[:, 1]

        # Apply baseline subtraction
        baseline = rolling_quantile_baseline(waveform, window_size, quantile)
        waveform_subtracted = waveform - baseline

        # Save result
        basename = os.path.basename(wf_file)
        output_file = os.path.join(output_dir, basename)
        np.savetxt(output_file, np.column_stack((time, waveform_subtracted)),
                   header="Time(ns)\tAmplitude", fmt="%.2f")

        print(f"✅ Baseline removed and saved: {output_file}")

# -------------------------------
# RUN SCRIPT
# -------------------------------
if __name__ == "__main__":
    subtract_baseline(
        input_dir="waveform_raw",
        output_dir="waveform_baseline_removed",
        window_size=31,
        quantile=0.1
    )


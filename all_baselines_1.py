import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import pandas as pd
import time as clock_timer
import glob
import os

# -------------------------------
# Rolling Quantile Baseline Extraction
# -------------------------------
def rolling_quantile_baseline(waveform, window_size=31, quantile=0.1):
    if window_size % 2 == 0:
        window_size += 1
    df = pd.Series(waveform)
    baseline = df.rolling(window=window_size, center=True, min_periods=1).quantile(quantile)
    signal = waveform - baseline.values
    return baseline.values, signal

# -------------------------------
# PEAK DETECTION
# -------------------------------
def extract_peaks(signal, height_threshold=2.0, distance=5):
    peaks, properties = find_peaks(signal, height=height_threshold, distance=distance)
    return peaks, properties

# -------------------------------
# CREATE OUTPUT PLOTS DIRECTORY
# -------------------------------
plots_dir = "plots"
os.makedirs(plots_dir, exist_ok=True)

# -------------------------------
# PROCESS ALL WAVEFORM FILES
# -------------------------------
waveform_files = sorted(glob.glob("waveform_output/waveform_*.txt"))

for wf_file in waveform_files:
    wf_name = os.path.splitext(os.path.basename(wf_file))[0]
    print(f"\n📂 Processing file: {wf_name}")

    # LOAD waveform
    waveform_data = np.loadtxt(wf_file, skiprows=1)
    time = waveform_data[:, 0]
    waveform = waveform_data[:, 1]

    # CLOCK START
    start_time = clock_timer.time()

    # Baseline extraction + peak detection
    baseline, signal = rolling_quantile_baseline(waveform, window_size=31, quantile=0.1)
    peaks, props = extract_peaks(signal, height_threshold=2.0, distance=5)

    # CLOCK END

    # PLOT AND SAVE TO FILE
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    axes[0].plot(time, waveform, label='Original waveform', linewidth=1.5)
    axes[0].plot(time, baseline, label='Rolling Quantile Baseline', linewidth=2)
    axes[0].set_title(f"{wf_name} - Baseline Extraction")
    axes[0].set_ylabel("ADC Amplitude")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(time, signal, label='Extracted Signal', linewidth=1.5)
    axes[1].plot(time[peaks], signal[peaks], 'rx', label='Detected Peaks', markersize=8, markeredgewidth=2)
    axes[1].set_title("Signal Extraction and Peak Detection")
    axes[1].set_xlabel("Time (ns)")
    axes[1].set_ylabel("Amplitude Above Baseline")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()

    plot_path = os.path.join(plots_dir, f"{wf_name}_plot.png")
    plt.savefig(plot_path)
    plt.close()

    end_time = clock_timer.time()
    elapsed_time = end_time - start_time

    # PRINT RESULTS
    print("Detected Peaks:")
    for i, idx in enumerate(peaks, 1):
        print(f"  Peak {i}: t0 = {time[idx]:.2f} ns, Amplitude = {signal[idx]:.2f}")
    print(f"⏱️  Time for extraction: {elapsed_time:.6f} seconds")
    print(f"📊 Plot saved: {plot_path}")

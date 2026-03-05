import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import pandas as pd
import time as clock_timer
import os

# -------------------------------
# LOAD WAVEFORM DATA
# -------------------------------
waveform_data = np.loadtxt("waveform_raw/waveform_0001.txt", skiprows=1)
time = waveform_data[:, 0]
waveform = waveform_data[:, 1]

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
# CLOCK START
# -------------------------------
start_time = clock_timer.time()

# Baseline extraction and peak detection
baseline, signal = rolling_quantile_baseline(waveform, window_size=31, quantile=0.1)
peaks, props = extract_peaks(signal, height_threshold=2.0, distance=5)

# CLOCK END
end_time = clock_timer.time()
elapsed_time = end_time - start_time

# -------------------------------
# PLOT RESULTS
# -------------------------------
fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

axes[0].plot(time, waveform, label='Original waveform', linewidth=1.5)
axes[0].plot(time, baseline, label='Rolling Quantile Baseline', linewidth=2)
axes[0].set_title("Rolling Quantile Baseline Extraction")
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
if os.environ.get("DISPLAY"):
    plt.show()
plt.close()

# -------------------------------
# PRINT RESULTS
# -------------------------------
print("\nDetected Peaks using Rolling Quantile Baseline:")
for i, idx in enumerate(peaks, 1):
    print(f"  Peak {i}: t0 = {time[idx]:.2f} ns, Amplitude = {signal[idx]:.2f}")

print(f"\n⏱️ Baseline extraction + peak detection time: {elapsed_time:.6f} seconds")

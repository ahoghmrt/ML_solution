import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# Reload data after kernel reset
true_data = np.load("ml_training_data/training_data_signals.npz")
pred_data = np.load("ml_training_data/predicted_signals.npz")

true_t0s = true_data["labels"][:, :, 0]
true_amps = true_data["labels"][:, :, 1]
pred_t0s = pred_data["t0s"]
pred_amps = pred_data["amps"]
presence = pred_data["presence"]

# Define bins (e.g., 10 ns width)
bins = np.linspace(0, 120, 13)
bin_centers = 0.5 * (bins[:-1] + bins[1:])
t0_resolution = []
amp_resolution = []
counts = []

# Analyze resolution in each bin
for i in range(len(bins) - 1):
    t0_errs, amp_errs = [], []
    for evt in range(true_t0s.shape[0]):
        for j in range(true_t0s.shape[1]):
            if presence[evt][j] < 0.5:
                continue
            t0 = true_t0s[evt][j]
            amp = true_amps[evt][j]
            if bins[i] <= t0 < bins[i + 1]:
                pred_t0 = pred_t0s[evt][j]
                pred_amp = pred_amps[evt][j]
                t0_errs.append(pred_t0 - t0)
                amp_errs.append(pred_amp - amp)
    if t0_errs:
        t0_resolution.append(np.std(t0_errs))
        amp_resolution.append(np.std(amp_errs))
        counts.append(len(t0_errs))
    else:
        t0_resolution.append(np.nan)
        amp_resolution.append(np.nan)
        counts.append(0)

# Display results
df = pd.DataFrame({
    "Bin Center (ns)": bin_centers,
    "t0 Resolution (ns)": t0_resolution,
    "Amplitude Resolution": amp_resolution,
    "Counts": counts
})

print("\nResolution vs t₀ bin:")
print(df.to_string(index=False))

import numpy as np
import os

# ----------------------------
# Configuration
# ----------------------------
max_signals = 5  # Maximum possible number of signals per waveform
input_dir = "waveform_baseline_removed"
truth_dir = "waveform_raw"
output_dir = "ml_training_data"
os.makedirs(output_dir, exist_ok=True)

waveforms = []
labels_regression = []  # (t0, A) padded up to max_signals
labels_count = []  # integer count of signals per waveform
time = None

for fname in sorted(os.listdir(input_dir)):
    if not fname.endswith(".txt"):
        continue

    wf_path = os.path.join(input_dir, fname)
    truth_fname = fname.replace("waveform", "truth")
    truth_path = os.path.join(truth_dir, truth_fname)

    waveform = np.loadtxt(wf_path)[:, 1]  # Load only amplitude column
    waveforms.append(waveform)
    if time is None:
        time = np.loadtxt(wf_path)[:, 0]  # Load time column once

    # Read truth file
    t0_amp_pairs = []
    with open(truth_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                try:
                    t0 = float(parts[1])
                    amp = float(parts[2])
                    t0_amp_pairs.append([t0, amp])
                except ValueError:
                    continue

    count = len(t0_amp_pairs)
    labels_count.append(count)

    # Pad t0_amp_pairs to max_signals
    t0_amp_padded = t0_amp_pairs[:max_signals] + [[0.0, 0.0]] * (max_signals - len(t0_amp_pairs))
    labels_regression.append(t0_amp_padded)

# Convert to arrays
waveforms = np.array(waveforms)
labels_regression = np.array(labels_regression)  # shape (samples, max_signals, 2)
labels_count = np.array(labels_count)  # shape (samples,)
time = np.array(time)

# Save separate datasets
np.savez(os.path.join(output_dir, "training_data_signals.npz"), waveforms=waveforms, labels=labels_regression, time=time)
np.savez(os.path.join(output_dir, "training_data_counts.npz"), waveforms=waveforms, labels=labels_count, time=time)

print("✅ Saved:")
print(" - training_data_signals.npz (for t0, amplitude regression)")
print(" - training_data_counts.npz (for signal count classification)")

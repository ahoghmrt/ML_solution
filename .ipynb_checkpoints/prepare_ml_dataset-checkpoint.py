import numpy as np
import os
from glob import glob

# -------------------------------
# Parameters
# -------------------------------
input_dir = "waveform_baseline_removed"
truth_dir = "waveform_raw"  # truth files still in original dir
output_file = "ml_training_data/training_data.npz"
max_signals = 5

os.makedirs("ml_training_data", exist_ok=True)

# -------------------------------
# Load waveforms and corresponding truth
# -------------------------------
waveform_files = sorted(glob(os.path.join(input_dir, "waveform_*.txt")))

all_waveforms = []
all_labels = []
time = None

for wf_file in waveform_files:
    # Load waveform
    data = np.loadtxt(wf_file, skiprows=1)
    time = data[:, 0] if time is None else time
    waveform = data[:, 1]
    all_waveforms.append(waveform)

    # Load truth
    truth_file = wf_file.replace("waveform_baseline_removed", "waveform_raw").replace("waveform", "truth")
    label_array = np.zeros((max_signals, 2))
    if os.path.exists(truth_file):
        truth_data = np.loadtxt(truth_file, skiprows=1)
        for i, (t0, amp) in enumerate(truth_data[:, 1:]):
            if i < max_signals:
                label_array[i, 0] = t0
                label_array[i, 1] = amp
    all_labels.append(label_array)

# -------------------------------
# Save dataset to .npz
# -------------------------------
all_waveforms = np.array(all_waveforms)
all_labels = np.array(all_labels)
np.savez(output_file, waveforms=all_waveforms, labels=all_labels, time=time)

print(f"✅ Saved ML training dataset: {output_file}")


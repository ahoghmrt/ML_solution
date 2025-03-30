import numpy as np
import os

# ----------------------------
# Configuration
# ----------------------------
max_signals = 5
input_dir = "waveform_baseline_removed"
truth_dir = "waveform_raw"
output_dir = "ml_training_data"
os.makedirs(output_dir, exist_ok=True)

waveforms = []
labels_regression = []  # (t0, A) padded up to max_signals
labels_count = []       # integer count of signals
candidate_t0s = []
candidate_amps = []
time = None

for fname in sorted(os.listdir(input_dir)):
    if not fname.endswith(".txt"):
        continue

    wf_path = os.path.join(input_dir, fname)
    truth_fname = fname.replace("waveform", "truth")
    truth_path = os.path.join(truth_dir, truth_fname)

    waveform = np.loadtxt(wf_path)[:, 1]  # amplitude
    waveforms.append(waveform)
    if time is None:
        time = np.loadtxt(wf_path)[:, 0]  # time (load once)

    # ----------------------------
    # Parse truth info: t0 and amplitude
    # ----------------------------
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

    # Pad or truncate to max_signals
    padded = t0_amp_pairs[:max_signals] + [[0.0, 0.0]] * (max_signals - len(t0_amp_pairs))
    labels_regression.append(padded)

    # Separate padded t0 and amp into two arrays
    t0_array = [pair[0] for pair in padded]
    amp_array = [pair[1] for pair in padded]
    candidate_t0s.append(t0_array)
    candidate_amps.append(amp_array)

# Convert to NumPy arrays
waveforms = np.array(waveforms)
labels_regression = np.array(labels_regression)       # (samples, max_signals, 2)
labels_count = np.array(labels_count)                 # (samples,)
candidate_t0s = np.array(candidate_t0s)               # (samples, max_signals)
candidate_amps = np.array(candidate_amps)             # (samples, max_signals)
time = np.array(time)

# ----------------------------
# Save to disk
# ----------------------------
np.savez(os.path.join(output_dir, "training_data_signals.npz"),
         waveforms=waveforms, labels=labels_regression, time=time)

np.savez(os.path.join(output_dir, "training_data_counts.npz"),
         waveforms=waveforms,
         labels=labels_count,
         candidate_t0s=candidate_t0s,
         candidate_amps=candidate_amps,
         time=time)

print("✅ Saved:")
print(" - training_data_signals.npz (for t0, amplitude regression)")
print(" - training_data_counts.npz (for signal count classification + signal candidates)")

# ----------------------------
# Verify structure
# ----------------------------
print("\n🔍 Verifying training_data_signals.npz:")
ds = np.load(os.path.join(output_dir, "training_data_signals.npz"))
print("  ✅ waveforms:", ds["waveforms"].shape)
print("  ✅ labels (t0, A):", ds["labels"].shape)
print("  ✅ time:", ds["time"].shape)

print("\n🔍 Verifying training_data_counts.npz:")
dc = np.load(os.path.join(output_dir, "training_data_counts.npz"))
print("  ✅ waveforms:", dc["waveforms"].shape)
print("  ✅ labels (counts):", dc["labels"].shape)
print("  ✅ candidate_t0s:", dc["candidate_t0s"].shape)
print("  ✅ candidate_amps:", dc["candidate_amps"].shape)
print("  ✅ time:", dc["time"].shape)

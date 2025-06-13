import numpy as np

# Load the signal regression dataset
data = np.load("ml_training_data/training_data_signals.npz")
labels = data["labels"]  # shape: (samples, max_signals, 2)

# Count how many non-zero amplitude entries exist per sample
signal_counts = np.sum(labels[:, :, 1] > 0, axis=1)  # axis=1: across signal slots

# Count unique values
unique, counts = np.unique(signal_counts, return_counts=True)

# Print result
print("📊 Signal count distribution (from amplitude > 0):")
for u, c in zip(unique, counts):
    print(f"  {int(u)} signals → {c} samples")

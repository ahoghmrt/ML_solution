import numpy as np
import os
import matplotlib.pyplot as plt
from glob import glob

# -------------------------------
# Plot waveform (with optional truth)
# -------------------------------
def plot_waveform(waveform_file, truth_file=None, save_plot=False, output_dir="plots"):
    # Load waveform
    data = np.loadtxt(waveform_file, skiprows=1)
    time = data[:, 0]
    amplitude = data[:, 1]

    # Start plot
    plt.figure(figsize=(12, 5))
    plt.plot(time, amplitude, label="Waveform", linewidth=1.5)

    # Load and plot truth
    if truth_file and os.path.exists(truth_file):
        truth_data = np.loadtxt(truth_file, skiprows=1)
        for t0, amp in truth_data[:, 1:]:
            if amp > 0:
                plt.plot(t0, amp, 'go', label="True Signal" if 'True Signal' not in plt.gca().get_legend_handles_labels()[1] else "")

    plt.title(f"Waveform: {os.path.basename(waveform_file)}")
    plt.xlabel("Time (ns)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Save plot if needed
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, os.path.basename(waveform_file).replace(".txt", ".png"))
    plt.savefig(plot_path)
    print(f"✅ Saved plot: {plot_path}")
    plt.close()

# -------------------------------
# RUN EXAMPLES
# -------------------------------
if __name__ == "__main__":
    input_dir = "waveform_baseline_removed"  # or "waveform_baseline_removed"
    waveform_files = sorted(glob(os.path.join(input_dir, "waveform_*.txt")))

    for i in range(1000):  # Show first 3 waveforms
        wf = waveform_files[i]
        truth = wf.replace("waveform", "truth").replace("waveform_raw", "waveform_raw")  # adjust path if needed
        plot_waveform(wf, truth_file=truth, save_plot=True)

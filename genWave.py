import numpy as np
import os
import random

# -------------------------------
# GLOBAL PARAMETERS
# -------------------------------
time_start = 0       # ns
time_end = 120       # ns
sampling_rate = 1    # ns (1 GHz)
n_samples = int((time_end - time_start) / sampling_rate)
time = np.linspace(time_start, time_end, n_samples)

# -------------------------------
# SIGNAL PULSE SHAPE FUNCTION
# -------------------------------
def pulse_shape(t, t0, amplitude, tau_rise=2, tau_fall=10):
    pulse = amplitude * (1 - np.exp(-(t - t0) / tau_rise)) * np.exp(-(t - t0) / tau_fall)
    pulse[t < t0] = 0
    return pulse

# -------------------------------
# WAVEFORM GENERATOR FUNCTION
# -------------------------------
def generate_waveform(n_signals=None, noise_std=0.5, min_spacing=10, random_seed=None, baseline=0.0):
    if random_seed is not None:
        np.random.seed(random_seed)
        random.seed(random_seed)

    if n_signals is None:
        n_signals = random.choice([2, 3, 4, 5])

    waveform = np.zeros_like(time)
    signal_truth = []

    t0_list = []
    attempts = 0
    while len(t0_list) < n_signals and attempts < 1000:
        candidate_t0 = np.random.uniform(5, time_end - 10)
        if all(abs(candidate_t0 - existing) >= min_spacing for existing in t0_list):
            t0_list.append(candidate_t0)
        attempts += 1

    t0_list.sort()

    for t0 in t0_list:
        amplitude = np.random.uniform(5, 20)
        signal = pulse_shape(time, t0, amplitude)
        waveform += signal
        realistic_amp = np.max(signal)  # more realistic amplitude after shaping
        signal_truth.append((t0, realistic_amp))

    noise = np.random.normal(0, noise_std, size=waveform.shape)
    waveform += noise + baseline

    return waveform, signal_truth, len(signal_truth)

# -------------------------------
# MAIN GENERATOR LOOP
# -------------------------------
def generate_dataset(num_waveforms=1000, output_dir="waveform_raw", noise_std=0.5, baseline=200.0, min_spacing=10):
    os.makedirs(output_dir, exist_ok=True)
    for i in range(num_waveforms):
        waveform, signal_truth, n_signals = generate_waveform(
            n_signals=None,
            noise_std=noise_std,
            min_spacing=min_spacing,
            random_seed=None,
            baseline=baseline
        )

        # Save waveform
        wf_file = os.path.join(output_dir, f"waveform_{i+1:04d}.txt")
        np.savetxt(wf_file, np.column_stack((time, waveform)), header="Time(ns)\tAmplitude", fmt="%.2f")

        # Save truth with t0 and realistic amplitude
        truth_file = os.path.join(output_dir, f"truth_{i+1:04d}.txt")
        with open(truth_file, "w") as f:
            f.write("Signal Index\tTime (ns)\tAmplitude\n")
            for idx, (t0, amp) in enumerate(signal_truth, 1):
                f.write(f"{idx}\t{t0:.2f}\t{amp:.2f}\n")
            f.write(f"Number of Signals: {n_signals}\n")

        print(f"✅ Saved: {wf_file}, {truth_file}")
        

# -------------------------------
# RUN SCRIPT
# -------------------------------
if __name__ == "__main__":
    generate_dataset(
        num_waveforms=100000,
        output_dir="waveform_raw",
        noise_std=0.3,
        baseline=200.0,
        min_spacing=2
    )

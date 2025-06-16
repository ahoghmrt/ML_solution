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
def generate_waveform(n_signals=None, noise_std=None, min_spacing=10, random_seed=None, baseline=0.0):
    if random_seed is not None:
        np.random.seed(random_seed)
        random.seed(random_seed)

    if noise_std is None:
        noise_std = random.choice([0.2, 0.3, 0.4, 0.5])

    if n_signals is None:
        n_signals = random.choice([0,0,1])

    waveform = np.zeros_like(time)
    signal_truth = []

    step_size = min_spacing
    possible_t0s = np.arange(5, time_end - 5, step_size)
    np.random.shuffle(possible_t0s)
    t0_list = sorted(possible_t0s[:n_signals])


    for t0 in t0_list:
        amplitude = np.random.uniform(5, 20)
        signal = pulse_shape(time, t0, amplitude)
        waveform += signal

        #realistic_amp = np.max(signal)  # more realistic amplitude after shaping
        peak_index = np.argmax(signal)
        peak_time = time[peak_index]
        realistic_amp = signal[peak_index]

        signal_truth.append((peak_time, realistic_amp))

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
        noise_std=0.5,
        baseline=200.0,
        min_spacing=0.001
    )

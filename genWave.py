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
        n_signals = random.choice([0,1,2,3])

    waveform = np.zeros_like(time)
    signal_truth = []

    step_size = min_spacing
    possible_t0s = np.arange(5, time_end - 5, step_size)
    np.random.shuffle(possible_t0s)
    t0_list = sorted(possible_t0s[:n_signals])

    tau_rise = np.random.uniform(2.0, 3.0)   # Rise time in ns
    tau_fall = np.random.uniform(10.0, 15.0)  # Fall time in ns

    main_pulses = []

    for t0 in t0_list:
        amplitude = np.random.uniform(5, 20)
        signal = pulse_shape(time, t0, amplitude, tau_rise, tau_fall)
        waveform += signal

        #realistic_amp = np.max(signal)  # more realistic amplitude after shaping
        peak_index = np.argmax(signal)
        peak_time = time[peak_index]
        realistic_amp = signal[peak_index]

       # signal_truth.append((peak_time, realistic_amp))
        signal_truth.append((t0, amplitude))
        main_pulses.append((t0, amplitude))


    # -------------------------------------
        # Undershoot: small negative dip after the main pulse
        # -------------------------------------
        undershoot_prob = 0.5  # 50% of signals have undershoot
        if np.random.rand() < undershoot_prob:
            undershoot_delay = np.random.uniform(2, 10)  # Delay after main peak
            undershoot_t0 = t0 + undershoot_delay
            if undershoot_t0 < time_end - 5:
                undershoot_amp = -amplitude * np.random.uniform(0.05, 0.2)
                undershoot = pulse_shape(time, undershoot_t0, abs(undershoot_amp), tau_rise=2, tau_fall=6)
                waveform -= undershoot  # Subtract to simulate undershoot

    noise = np.random.normal(0, noise_std, size=waveform.shape)
    waveform += noise + baseline

    # -------------------------------------
    # Afterpulses
    # -------------------------------------
    afterpulse_probability = 0.3  # 30% chance each signal has an afterpulse
    reflection_probability = 0.3  # 30% chance each signal has a reflection
    reflection_delay = 20  # ns fixed delay for reflection

    for t0, amp in main_pulses:
        # Afterpulse
        if np.random.rand() < afterpulse_probability:
            delay = np.random.uniform(10, 40)  # random delay
            after_t0 = t0 + delay
            if after_t0 < time_end - 5:
                after_amp = amp * np.random.uniform(0.01, 0.1)
                after_signal = pulse_shape(time, after_t0, after_amp)
                waveform += after_signal

        # Reflection peak
        if np.random.rand() < reflection_probability:
            n_echoes = np.random.randint(1, 4)  # 1–3 reflections
            for i in range(n_echoes):
                refl_delay = np.random.uniform(15, 40) * (i + 1)  # increasing delay
                refl_t0 = t0 + refl_delay
                if refl_t0 >= time_end - 5:
                    break  # outside waveform window

                refl_amp = amp * (0.05 * (0.5 ** i))  # decaying: e.g., 5%, 2.5%, ...
                refl_signal = pulse_shape(time, refl_t0, refl_amp, tau_rise, tau_fall)
                waveform += refl_signal

    return waveform, signal_truth, len(signal_truth)

# -------------------------------
# MAIN GENERATOR LOOP
# -------------------------------
def generate_dataset(num_waveforms=1000, output_dir="waveform_raw", noise_std=0.5, baseline=200.0, min_spacing=10):
    os.makedirs(output_dir, exist_ok=True)
    for i in range(num_waveforms):
        random_baseline = np.random.uniform(100, 400) 
        waveform, signal_truth, n_signals = generate_waveform(
            n_signals=None,
            noise_std=noise_std,
            min_spacing=min_spacing,
            random_seed=None,
            baseline=random_baseline
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
        num_waveforms=40000,
        output_dir="waveform_raw",
        noise_std=0.5,
        baseline=200.0,
        min_spacing=1
    )

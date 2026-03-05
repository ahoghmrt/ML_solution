import numpy as np
import os
import random
import logging
import config as cfg

logger = logging.getLogger(__name__)

# -------------------------------
# GLOBAL PARAMETERS
# -------------------------------
n_samples = int((cfg.TIME_END - cfg.TIME_START) / cfg.SAMPLING_RATE)
time = np.linspace(cfg.TIME_START, cfg.TIME_END, n_samples)

# -------------------------------
# SIGNAL PULSE SHAPE FUNCTION
# -------------------------------
def pulse_shape(t, t0, amplitude, tau_rise=cfg.TAU_RISE, tau_fall=cfg.TAU_FALL):
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
        noise_std = random.choice(cfg.NOISE_STD_CHOICES)

    if n_signals is None:
        n_signals = random.choice(cfg.SIGNAL_COUNTS)

    waveform = np.zeros_like(time)
    signal_truth = []

    t0_list = []
    attempts = 0
    while len(t0_list) < n_signals and attempts < 1000:
        candidate_t0 = np.random.uniform(cfg.T0_MIN, cfg.TIME_END - cfg.T0_MARGIN)
        if all(abs(candidate_t0 - existing) >= min_spacing for existing in t0_list):
            t0_list.append(candidate_t0)
        attempts += 1

    t0_list.sort()

    for t0 in t0_list:
        amplitude = np.random.uniform(cfg.AMPLITUDE_MIN, cfg.AMPLITUDE_MAX)
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
def generate_dataset(num_waveforms=cfg.NUM_WAVEFORMS, output_dir=cfg.DIR_RAW, noise_std=cfg.NOISE_STD, baseline=cfg.BASELINE, min_spacing=cfg.MIN_SPACING, max_signals=cfg.MAX_SIGNALS):
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Generating {num_waveforms} waveforms into '{output_dir}/'")

    all_waveforms = []
    all_truth = []
    all_counts = []

    for i in range(num_waveforms):
        waveform, signal_truth, n_signals = generate_waveform(
            n_signals=None,
            noise_std=noise_std,
            min_spacing=min_spacing,
            random_seed=None,
            baseline=baseline
        )

        all_waveforms.append(waveform)
        all_counts.append(n_signals)

        # Pad truth to max_signals
        padded = signal_truth[:max_signals] + [(0.0, 0.0)] * (max_signals - len(signal_truth))
        all_truth.append(padded)

        if (i + 1) % 100 == 0 or (i + 1) == num_waveforms:
            logger.debug(f"Generated {i + 1}/{num_waveforms} waveforms")

    np.savez(os.path.join(output_dir, "data.npz"),
             waveforms=np.array(all_waveforms),
             time=time,
             truth=np.array(all_truth),
             counts=np.array(all_counts))

    logger.info(f"Generated {num_waveforms} waveforms in '{output_dir}/'")

# -------------------------------
# RUN SCRIPT
# -------------------------------
if __name__ == "__main__":
    generate_dataset()

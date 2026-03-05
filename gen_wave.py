import numpy as np
import os
import logging
import config as cfg

logger = logging.getLogger(__name__)

# -------------------------------
# GLOBAL PARAMETERS
# -------------------------------
n_samples = int((cfg.TIME_END - cfg.TIME_START) / cfg.SAMPLING_RATE)
time = np.linspace(cfg.TIME_START, cfg.TIME_END, n_samples)


def _generate_t0s(group_size, k, min_spacing):
    """Generate sorted t0 arrays with min_spacing enforced, shape (group_size, k)."""
    t0_min, t0_max = cfg.T0_MIN, cfg.TIME_END - cfg.T0_MARGIN
    t0s = np.random.uniform(t0_min, t0_max, (group_size, k))
    t0s.sort(axis=1)
    if k > 1 and min_spacing > 0:
        # Retry rows that violate spacing
        diffs = np.diff(t0s, axis=1)
        bad = np.any(diffs < min_spacing, axis=1)
        retries = 0
        while np.any(bad) and retries < 100:
            n_bad = bad.sum()
            t0s[bad] = np.random.uniform(t0_min, t0_max, (n_bad, k))
            t0s[bad] = np.sort(t0s[bad], axis=1)
            diffs = np.diff(t0s, axis=1)
            bad = np.any(diffs < min_spacing, axis=1)
            retries += 1
    return t0s


def _vectorized_pulses(t0s, amps):
    """Compute pulse waveforms for a batch.

    t0s:  (batch, k)
    amps: (batch, k)
    Returns: waveforms (batch, n_samples), peak_times (batch, k), peak_amps (batch, k)
    """
    # Broadcasting: time[1, 1, T] - t0s[B, K, 1] → dt[B, K, T]
    dt = time[np.newaxis, np.newaxis, :] - t0s[:, :, np.newaxis]
    pulses = amps[:, :, np.newaxis] * (1 - np.exp(-dt / cfg.TAU_RISE)) * np.exp(-dt / cfg.TAU_FALL)
    pulses[dt < 0] = 0  # zero before t0

    # Peak extraction per signal
    peak_idx = np.argmax(pulses, axis=2)  # (batch, k)
    peak_times = time[peak_idx]
    peak_amps = np.take_along_axis(pulses, peak_idx[:, :, np.newaxis], axis=2).squeeze(2)

    # Sum across signals → waveform
    waveforms = pulses.sum(axis=1)  # (batch, n_samples)
    return waveforms, peak_times, peak_amps


# -------------------------------
# MAIN GENERATOR
# -------------------------------
def generate_dataset(num_waveforms=cfg.NUM_WAVEFORMS, output_dir=cfg.DIR_RAW, noise_std=cfg.NOISE_STD, baseline=cfg.BASELINE, min_spacing=cfg.MIN_SPACING, max_signals=cfg.MAX_SIGNALS, varied_noise=cfg.VARIED_NOISE):
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Generating {num_waveforms} waveforms into '{output_dir}/'")

    # Pre-draw signal counts for all waveforms
    counts = np.random.choice(cfg.SIGNAL_COUNTS, size=num_waveforms)

    # Pre-allocate outputs
    all_waveforms = np.empty((num_waveforms, len(time)))
    all_truth = np.zeros((num_waveforms, max_signals, 2))

    # Process each signal-count group in a vectorized batch
    for k in range(max(cfg.SIGNAL_COUNTS) + 1):
        mask = counts == k
        group_size = mask.sum()
        if group_size == 0:
            continue

        if k == 0:
            all_waveforms[mask] = 0.0
            continue

        t0s = _generate_t0s(group_size, k, min_spacing)
        amps = np.random.uniform(cfg.AMPLITUDE_MIN, cfg.AMPLITUDE_MAX, (group_size, k))

        waveforms, peak_times, peak_amps = _vectorized_pulses(t0s, amps)
        all_waveforms[mask] = waveforms

        # Store truth (peak_time, peak_amp) for each signal slot
        all_truth[mask, :k, 0] = peak_times
        all_truth[mask, :k, 1] = peak_amps

    # Add noise and baseline to all waveforms at once
    if varied_noise:
        noise_stds = np.random.choice(cfg.NOISE_STD_CHOICES, size=num_waveforms)
        noise = noise_stds[:, np.newaxis] * np.random.standard_normal(all_waveforms.shape)
        all_waveforms += noise + baseline
    else:
        all_waveforms += np.random.normal(0, noise_std, all_waveforms.shape) + baseline

    np.savez(os.path.join(output_dir, "data.npz"),
             waveforms=all_waveforms,
             time=time,
             truth=all_truth,
             counts=counts)

    logger.info(f"Generated {num_waveforms} waveforms in '{output_dir}/'")


if __name__ == "__main__":
    generate_dataset()

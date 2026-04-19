import numpy as np
import os
import logging
import config as cfg

logger = logging.getLogger(__name__)

# -------------------------------
# GLOBAL PARAMETERS
# -------------------------------
# Exact 1 ns/sample grid: time = [0, 1, 2, ..., TIME_END - 1] with SAMPLING_RATE=1.
# The previous np.linspace(0, 120, 120) gave ~1.008 ns/sample, which silently
# shifted every subsequent operation off the nominal grid.
time = np.arange(cfg.TIME_START, cfg.TIME_END, cfg.SAMPLING_RATE, dtype=float)


def _generate_t0s(group_size, k, min_spacing):
    """Generate sorted t0 arrays with min_spacing enforced, shape (group_size, k)."""
    t0_min, t0_max = cfg.T0_MIN, cfg.TIME_END - cfg.T0_MARGIN
    t0s = np.random.uniform(t0_min, t0_max, (group_size, k))
    t0s.sort(axis=1)
    if k > 1 and min_spacing > 0:
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
    """Compute the combined waveform for a batch of signals.

    t0s:  (batch, k)
    amps: (batch, k)  — pulse scale parameter
    Returns: waveforms (batch, n_samples)
    """
    dt = time[np.newaxis, np.newaxis, :] - t0s[:, :, np.newaxis]
    pulses = amps[:, :, np.newaxis] * (1 - np.exp(-dt / cfg.TAU_RISE)) * np.exp(-dt / cfg.TAU_FALL)
    pulses[dt < 0] = 0
    return pulses.sum(axis=1)


def _baseline_profiles(num_waveforms, baseline, jitter, drift_max):
    """Per-waveform non-constant baseline.

    Each waveform gets a random constant offset + a random linear drift across
    the window. This makes the baseline subtraction step actually do work
    (previously every waveform had an identical constant baseline, which made
    subtraction trivial and inflated confidence in that step).
    """
    offsets = np.random.uniform(-jitter, jitter, num_waveforms)  # (N,)
    slopes = np.random.uniform(-drift_max, drift_max, num_waveforms) / max(len(time) - 1, 1)  # (N,)
    centered = time - time.mean()  # (T,)
    return baseline + offsets[:, np.newaxis] + slopes[:, np.newaxis] * centered[np.newaxis, :]


# -------------------------------
# MAIN GENERATOR
# -------------------------------
def generate_dataset(
    num_waveforms=cfg.NUM_WAVEFORMS,
    output_dir=cfg.DIR_RAW,
    noise_std=cfg.NOISE_STD,
    baseline=cfg.BASELINE,
    min_spacing=cfg.MIN_SPACING,
    max_signals=cfg.MAX_SIGNALS,
    varied_noise=cfg.VARIED_NOISE,
    baseline_jitter=cfg.BASELINE_JITTER,
    baseline_drift_max=cfg.BASELINE_DRIFT_MAX,
):
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Generating {num_waveforms} waveforms into '{output_dir}/'")

    counts = np.random.choice(cfg.SIGNAL_COUNTS, size=num_waveforms)

    all_waveforms = np.zeros((num_waveforms, len(time)))
    # Truth stores the CONTINUOUS physics parameters: (t0 in ns, pulse-scale amp).
    # Previously stored the quantized peak time (time[argmax]) and standalone
    # per-pulse peak amplitude. Both were lossy: t0 was rounded to the sample
    # grid (capping achievable resolution at ~1 ns), and the stored amplitude
    # was a shape-dependent fraction (~0.58×) of the generator input, which
    # didn't match the AMPLITUDE_MIN/MAX range declared in config.
    all_truth = np.zeros((num_waveforms, max_signals, 2))

    for k in range(max(cfg.SIGNAL_COUNTS) + 1):
        mask = counts == k
        group_size = mask.sum()
        if group_size == 0:
            continue
        if k == 0:
            continue  # waveforms stay zero; truth stays zero

        t0s = _generate_t0s(group_size, k, min_spacing)
        amps = np.random.uniform(cfg.AMPLITUDE_MIN, cfg.AMPLITUDE_MAX, (group_size, k))

        all_waveforms[mask] = _vectorized_pulses(t0s, amps)
        all_truth[mask, :k, 0] = t0s
        all_truth[mask, :k, 1] = amps

    # Add noise
    if varied_noise:
        noise_stds = np.random.choice(cfg.NOISE_STD_CHOICES, size=num_waveforms)
        all_waveforms += noise_stds[:, np.newaxis] * np.random.standard_normal(all_waveforms.shape)
    else:
        all_waveforms += np.random.normal(0, noise_std, all_waveforms.shape)

    # Add non-trivial baseline (offset jitter + linear drift per waveform)
    all_waveforms += _baseline_profiles(num_waveforms, baseline, baseline_jitter, baseline_drift_max)

    np.savez(
        os.path.join(output_dir, "data.npz"),
        waveforms=all_waveforms,
        time=time,
        truth=all_truth,
        counts=counts,
    )

    logger.info(f"Generated {num_waveforms} waveforms in '{output_dir}/'")


if __name__ == "__main__":
    generate_dataset()

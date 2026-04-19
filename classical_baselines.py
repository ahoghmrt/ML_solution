"""Classical (non-ML) pulse extraction baselines.

Two methods, both operating on baseline-subtracted waveforms and returning
a list of (t0_in_ns, amplitude) pairs per waveform:

- matched_filter:   optimal linear filter for isolated pulses in white
                    Gaussian noise. Fails on overlap because neighbouring
                    pulses interfere in the filter output.
- iterative_clean:  CLEAN-like greedy deconvolution. Find the strongest
                    template-projection, subtract the predicted pulse from
                    the residual, repeat. Standard pile-up baseline; the
                    closest classical analogue to what the ML model learns.

The returned amplitudes are in the same units as the config's
AMPLITUDE_MIN/MAX (i.e. the pulse-scale parameter, not the visible peak),
so they are directly comparable to the truth labels.
"""
from __future__ import annotations
import numpy as np
from scipy.signal import find_peaks
import config as cfg


def _build_template(length: int = 60) -> np.ndarray:
    """Canonical bi-exponential pulse starting at t=0, amp=1."""
    t = np.arange(length, dtype=float)
    return (1 - np.exp(-t / cfg.TAU_RISE)) * np.exp(-t / cfg.TAU_FALL)


_TEMPLATE = _build_template()
_TEMPLATE_NORM = float(np.sum(_TEMPLATE ** 2))


def _projection_amps(residual: np.ndarray) -> np.ndarray:
    """Best-fit amplitude assuming a single pulse starts at each sample.

    projection[k] = <residual[k : k+T], template> / ||template||^2
    Peaks in projection = pulse start locations; the value at a peak is the
    maximum-likelihood amplitude for that position under white Gaussian noise.
    """
    corr = np.correlate(residual, _TEMPLATE, mode="valid")  # (L - T + 1,)
    return corr / _TEMPLATE_NORM


def matched_filter(waveform: np.ndarray, threshold: float = 2.0) -> list[tuple[float, float]]:
    """One-shot matched-filter peak detection (no overlap handling)."""
    amps = _projection_amps(waveform)
    min_dist = max(1, int(round(cfg.MIN_SPACING / cfg.SAMPLING_RATE)))
    peaks, _ = find_peaks(amps, height=threshold, distance=min_dist)
    return [(float(p * cfg.SAMPLING_RATE), float(amps[p])) for p in peaks]


def iterative_clean(
    waveform: np.ndarray,
    max_signals: int = max(cfg.SIGNAL_COUNTS),
    threshold: float = 2.0,
) -> list[tuple[float, float]]:
    """CLEAN-style iterative template subtraction."""
    residual = waveform.astype(float).copy()
    T = len(_TEMPLATE)
    found: list[tuple[float, float]] = []

    for _ in range(max_signals):
        amps = _projection_amps(residual)
        k = int(np.argmax(amps))
        best = float(amps[k])
        if best < threshold:
            break
        # Subtract the predicted pulse at sample position k
        end = min(len(residual), k + T)
        residual[k:end] -= best * _TEMPLATE[: end - k]
        found.append((float(k * cfg.SAMPLING_RATE), best))

    found.sort(key=lambda x: x[0])
    return found

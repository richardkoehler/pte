"""Module for burst analysis."""

import numpy as np


def burst_length_and_amplitude(
    power: np.ndarray, threshold: int | float, sfreq: int | float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculates the duration, amplitude and indexes of bursts.

    Args:
        power (np.ndarray): Power as array of shape (n_samples, )
        threshold (int | float): Threshold of power for identifying bursts
        sfreq (int | float): Sampling frequency

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]:
        Burst length (in seconds), mean burst amplitude and start and end
        indexes of individual bursts
    """
    bursts = np.zeros(power.shape[0] + 1)
    bursts[1:] = power >= threshold
    bursts_onoff = np.diff(bursts)
    is_burst = False
    burst_start = 0
    burst_lengths = []
    burst_amplitudes = []
    burst_indices = []
    for index, threshold_crossed in enumerate(bursts_onoff):
        if threshold_crossed:
            if is_burst:
                burst_lengths.append(index - burst_start)
                burst_amplitudes.append(power[burst_start:index].mean())
                burst_indices.append([burst_start, index])
                is_burst = False
            else:
                burst_start = index
                is_burst = True
    if is_burst:
        burst_lengths.append(index + 1 - burst_start)
        burst_amplitudes.append(power[burst_start : index + 1].mean())
        burst_indices.append([burst_start, index + 1])
    return (
        np.array(burst_lengths) / sfreq,
        np.array(burst_amplitudes),
        np.array(burst_indices),
    )

from typing import Union

import numpy as np


def burst_length_and_amplitude(
    power: np.ndarray,
    threshold: int | float,
    sfreq: int | float,
    return_burst_amplitude: bool = True,
    return_burst_indexes: bool = True,
) -> (
    np.ndarray |
    tuple[np.ndarray, np.ndarray] |
    tuple[np.ndarray, np.ndarray, np.ndarray]
):
    """Calculates the duration, amplitude and indexes of bursts.

    Args:
        power (np.ndarray): Power as array of shape (n_samples, )
        threshold (int | float): Threshold of power for identifying bursts
        sfreq (int | float): Sampling frequency
        return_burst_amplitude (bool, optional): If True, return burst
        amplitude. Defaults to True.
        return_burst_indexes (bool, optional): If True, return start and stop
        indexes of individual bursts. Defaults to True.

    Returns:
        np.ndarray |
        tuple[np.ndarray, np.ndarray] |
        tuple[np.ndarray, np.ndarray, np.ndarray]:
        Burst length (in seconds), mean burst amplitude and start and end
        indexes of individual bursts
    """
    bursts = np.zeros(power.shape[0] + 1)
    bursts[1:] = power >= threshold
    bursts_onoff = np.diff(bursts)
    isburst = False
    burst_length = []
    burst_start = 0
    burst_amplitude = []
    burst_indexes = []
    for index, threshold_crossed in enumerate(bursts_onoff):
        if threshold_crossed:
            if isburst:
                burst_length.append(index - burst_start)
                burst_amplitude.append(power[burst_start:index].mean())
                burst_indexes.append([burst_start, index])
                isburst = False
            else:
                burst_start = index
                isburst = True
    if isburst:
        burst_length.append(index + 1 - burst_start)
        burst_amplitude.append(power[burst_start : index + 1].mean())
        burst_indexes.append([burst_start, index + 1])
    burst_length = np.array(burst_length) / sfreq
    returns = [burst_length]
    if return_burst_amplitude:
        returns.append(np.array(burst_amplitude))
    if return_burst_indexes:
        returns.append(np.array(burst_indexes))
    return tuple(returns)

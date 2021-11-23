"""Module to handle events."""

from typing import Union

import numpy as np
from numba import njit


@njit
def threshold_events(
    data: np.ndarray, threshold: Union[float, int]
) -> np.ndarray:
    """Apply threshold to find start and end of events.

    Arguments
    ---------
    data : np.ndarray
        Input data to apply thresholding to.
    threshold : float | int
        Threshold value.

    Returns
    -------
    np.ndarray
        Event array.
    """

    onoff = np.where(data > threshold, 1, 0)
    onoff_diff = np.zeros_like(onoff)
    onoff_diff[1:] = np.diff(onoff)
    index_start = np.where(onoff_diff == 1)[0]
    index_stop = np.where(onoff_diff == -1)[0]
    arr_start = np.stack(
        (index_start, np.zeros_like(index_start), np.ones_like(index_start)),
        axis=1,
    )
    arr_stop = np.stack(
        (index_stop, np.zeros_like(index_stop), np.ones_like(index_stop) * -1),
        axis=1,
    )
    return np.vstack((arr_start, arr_stop))

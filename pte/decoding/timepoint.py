"""Module for calculating earliest significant time point of prediction."""
from typing import Iterable, Optional, Union
import numpy as np

import pte


def get_earliest_timepoint(
    data: np.ndarray,
    x_lims: tuple,
    sfreq: Optional[Union[int, float]] = None,
    threshold: Union[
        int, float, tuple[Union[int, float], Union[int, float]]
    ] = (0.0, 1.0),
    n_perm: int = 1000,
    alpha: float = 0.05,
    correction_method: str = "cluster",
):
    """Get earliest timepoint of motor onset prediction."""
    threshold_value, _ = transform_threshold(
        threshold=threshold, sfreq=sfreq, data=data.T
    )

    p_vals = pte.stats.timeseries_pvals(
        x=data.T, y=threshold_value, n_perm=n_perm, two_tailed=False
    )

    clusters, cluster_count = pte.stats.clusters_from_pvals(
        p_vals=p_vals,
        alpha=alpha,
        correction_method=correction_method,
        n_perm=n_perm,
    )

    if cluster_count > 0:
        x_labels = np.linspace(x_lims[0], x_lims[1], len(p_vals)).round(2)
        index = np.where(clusters != 0)[0][0]
        return x_labels[index]
    return None


def transform_threshold(
    threshold: Union[int, float, Iterable],
    sfreq: Union[int, float],
    data: np.ndarray,
):
    """Take threshold input and return threshold value and array."""
    if isinstance(threshold, (int, float)):
        threshold_value = threshold
    else:
        threshold_value = np.mean(
            data[int(threshold[0] * sfreq) : int(threshold[1] * sfreq)]
        )
    threshold_arr = np.ones(data.shape[0]) * threshold_value
    return threshold_value, threshold_arr

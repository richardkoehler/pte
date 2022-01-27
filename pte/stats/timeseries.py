"""Module for functions regarding multiple comparison."""
from typing import Optional, Union

import numpy as np
from statsmodels.stats.multitest import fdrcorrection

import pte


def timeseries_pvals(
    x: np.ndarray,
    y: Union[int, float, np.ndarray],
    n_perm: int,
    two_tailed: bool,
):
    """Calculate sample-wise p-values for array of predictions."""
    p_vals = np.empty(len(x))
    if isinstance(y, (int, float)):
        for t_p, pred in enumerate(x):
            _, p_vals[t_p] = pte.stats.permutation_onesample(
                x=pred, y=y, n_perm=n_perm, two_tailed=two_tailed
            )
    else:
        for i, (x_, y_) in enumerate(zip(x, y)):
            _, p_vals[i] = pte.stats.permutation_twosample(
                x=x_, y=y_, n_perm=n_perm, two_tailed=two_tailed
            )
    return p_vals


def correct_pvals(
    p_vals: np.ndarray,
    alpha: float = 0.05,
    correction_method: str = "cluster",
    n_perm: Optional[int] = 10000,
):
    """Correct p-values for multiple comparisons."""
    if correction_method == "cluster":
        _, signif = pte.stats.clusterwise_pval_numba(
            p_values=p_vals, alpha=alpha, n_perm=n_perm, only_max_cluster=False
        )
        if len(signif) > 0:
            signif = np.hstack(signif)
        else:
            signif = np.array([])
    elif correction_method == "fdr":
        rejected, _ = fdrcorrection(
            pvals=p_vals, alpha=alpha, method="poscorr", is_sorted=False
        )
        signif = np.where(rejected)[0]
    else:
        raise ValueError(
            f"`correction_method` must be one of either `cluster` or `fdr`. \
            Got:{correction_method}."
        )
    return signif

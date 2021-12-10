"""Module for plotting decoding results."""
import os
from pathlib import Path
from typing import Iterable, Optional, Union

import matplotlib
import numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt

import pte


def lineplot_prediction(
    ecog_data: np.ndarray,
    lfp_data: np.ndarray,
    subpl_titles: Iterable,
    sfreq: Union[int, float],
    x_lims: tuple,
    fpath: Optional[Union[Path, str]] = None,
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    label: str = "Distance from Hyperplane",
    threshold: Union[int, float, tuple] = (0.0, 1.0),
    p_lim: float = 0.05,
    n_perm: int = 1000,
    ylims=None,
) -> None:
    """"""
    viridis = cm.get_cmap("viridis", 8)
    colors = viridis(4), viridis(2)

    fig, axs = plt.subplots(ncols=1, nrows=2, figsize=(7, 5), sharex=False)

    n_samples = ecog_data.shape[0]

    for i, data in enumerate((ecog_data, lfp_data)):
        _single_lineplot(
            data=data,
            ax=axs[i],
            threshold=threshold,
            sfreq=sfreq,
            color=colors[i],
            subpl_title=subpl_titles[i],
            label=label,
            p_lim=p_lim,
            n_perm=n_perm,
        )

    for ax in axs:
        ax.legend(loc="upper left", fontsize="small")
        if ylims:
            ax.set_ylim(ylims[0], ylims[1])
        xticks = np.arange(0, n_samples, sfreq)
        ax.set_xticks(xticks)
        ax.set_xticklabels(np.linspace(x_lims[0], x_lims[1], len(xticks)))
        ax.set_xlabel("Time (s)")

    if subtitle:
        fig.text(
            x=0.5,
            y=0.94,
            s=subtitle,
            fontsize="medium",
            ha="center",
            style="italic",
            transform=fig.transFigure,
        )
    fig.suptitle(title, fontsize="large", y=1.01)
    fig.tight_layout()
    if fpath:
        fig.savefig(os.path.normpath(fpath), bbox_inches="tight", dpi=300)
    plt.show(block=True)


def _single_lineplot(
    data: np.ndarray,
    ax: matplotlib.axes.Axes,
    threshold: Union[Iterable, int, float],
    sfreq: Union[int, float],
    color: tuple,
    label: str,
    subpl_title: Optional[str],
    p_lim: float,
    n_perm: Optional[int],
):
    """Plot prediction line for single model."""
    viridis = cm.get_cmap("viridis", 8)
    c_signif = viridis(7)
    if isinstance(threshold, (int, float)):
        threshold_value = threshold
        threshold_arr = np.ones_like(data[:, 0]) * threshold_value
    else:
        threshold_value = data[
            int(threshold[0] * sfreq) : int(threshold[1] * sfreq)
        ].mean()
        threshold_arr = np.ones_like(data[:, 0]) * threshold_value
    p_vals = np.empty(data.shape[0])
    for t_p, pred in enumerate(data):
        _, p_vals[t_p] = pte.stats.permutation_numba_onesample(
            x=pred, y=threshold_value, n_perm=n_perm, two_tailed=False
        )
    ax.plot(data.mean(axis=1), color=color, label=label)
    ax.plot(
        threshold_arr,
        color="r",
        label="Threshold",
        alpha=0.5,
        linestyle="dashed",
    )
    ax.fill_between(
        np.arange(data.shape[0]),
        data.mean(axis=1) - data.std(axis=1),
        data.mean(axis=1) + data.std(axis=1),
        alpha=0.5,
        color=color,
    )
    signif = np.where(p_vals <= p_lim)[0]
    if signif.size > 0:
        _, signif = pte.stats.clusterwise_pval_numba(
            p_arr=p_vals, p_sig=p_lim, n_perm=n_perm
        )
        if len(signif) == 0:
            pass
        elif len(signif) == 1:
            signif = np.hstack(signif)
            lims = np.arange(signif[0], signif[-1])
            ax.fill_between(
                lims,
                data.mean(axis=1)[lims],
                threshold_arr[lims],
                alpha=0.5,
                color=c_signif,
                label=f"p-value <= {p_lim}",
            )
        else:
            for i, sig in enumerate(signif):
                lims = np.arange(sig[0], sig[-1])
                label = f"p-value <= {p_lim}" if i == 0 else None
                ax.fill_between(
                    lims,
                    data.mean(axis=1)[lims],
                    threshold_arr(axis=1)[lims],
                    alpha=0.5,
                    color=c_signif,
                    label=label,
                )
    ax.set_title(subpl_title, fontsize="medium")

"""Module for plotting clusters."""
from pathlib import Path
from typing import Literal, Sequence

import matplotlib.figure
from matplotlib import pyplot as plt
import numpy as np
import pte_stats
import scipy.stats


def clusterplot_correlation(
    data: np.ndarray,
    corr_data: np.ndarray,
    extent: tuple | list,
    alpha: float = 0.05,
    n_perm: int = 100,
    fig: matplotlib.figure.Figure | None = None,
    title: str | None = None,
    x_label: str = "Time [s]",
    y_label: str = "Frequency [Hz]",
    cbar_label: str = "Power [AU]",
    cbar_borderval: str | int | float = "auto",
    cbar_borderval_corr: str | int | float = "auto",
    outpath: Path | str | None = None,
    show: bool = True,
    n_jobs: int = 1,
) -> matplotlib.figure.Figure:
    """Plot power, p-values and significant clusters."""
    data_av = data.mean(axis=0)

    if isinstance(cbar_borderval, str):
        if cbar_borderval != "auto":
            raise ValueError(
                "`cbar_borderval` must be either an int, float or"
                f" 'auto'. Got: {cbar_borderval}."
            )
        cbar_borderval: float = min(data_av.max(), np.abs(data_av.min()))

    if not fig:
        fig, axs = plt.subplots(
            nrows=1,
            ncols=3,
            figsize=(8, 2.4),
            sharex=True,
            sharey=True,
        )
    else:
        axs = fig.axes
    for ax in axs:
        ax.set_xlabel(x_label)
    axs[0].set_ylabel(y_label)

    corr_vals = np.empty(data.shape[1:])
    p_values = np.empty(data.shape[1:])
    for i in range(data.shape[1]):
        for j in range(data.shape[2]):
            corr, pval = scipy.stats.spearmanr(data[:, i, j], corr_data)
            corr_vals[i, j] = corr
            p_values[i, j] = pval

    _, cluster_arr = pte_stats.cluster_correct_pvals_2d(
        p_values=p_values,
        alpha=alpha,
        n_perm=n_perm,
        only_max_cluster=False,
        n_jobs=n_jobs,
    )
    squared = np.zeros(data.shape[1:], dtype=int)
    if cluster_arr:
        for cluster in cluster_arr:
            if cluster[0].size > 20:
                squared[cluster] = 1
        axs[0].contour(
            squared,
            levels=[0.99],
            extent=extent,
            origin="lower",
            colors="black",
        )
        axs[1].contour(
            squared,
            levels=[0.99],
            extent=extent,
            origin="lower",
            colors="black",
        )

    pos_0 = axs[0].imshow(
        data_av,
        extent=extent,
        cmap="viridis",
        aspect="auto",
        origin="lower",
        vmin=cbar_borderval * -1,
        vmax=cbar_borderval,
    )
    fig.colorbar(pos_0, ax=axs[0], label=cbar_label)

    if cbar_borderval_corr == "auto":
        cbar_borderval_corr: float = min(
            corr_vals.max(), np.abs(corr_vals.min())
        )
    corr_vals_masked = np.ma.masked_where(np.logical_not(squared), corr_vals)
    pos_1 = axs[1].imshow(
        corr_vals_masked,
        extent=extent,
        cmap="viridis",
        aspect="auto",
        origin="lower",
        vmax=cbar_borderval_corr,
        vmin=-cbar_borderval_corr,
    )
    fig.colorbar(pos_1, ax=axs[1], label="Spearman's ρ")

    pos_2 = axs[2].imshow(
        p_values,
        extent=extent,
        norm="log",  # type: ignore
        cmap="viridis_r",
        aspect="auto",
        origin="lower",
    )
    axs[2].contour(
        p_values,
        levels=[alpha],
        extent=extent,
        origin="lower",
        colors="black",
    )
    fig.colorbar(pos_2, ax=axs[2], label="P [log]")

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    if outpath:
        fig.savefig(outpath)
    if show:
        plt.show()
    return fig


def clusterplot_combined(
    power_a: np.ndarray,
    power_b: np.ndarray | int | float,
    extent: tuple | list,
    alpha: float = 0.05,
    n_perm: int = 1000,
    correction_method: Literal["cluster_pvals", "cluster"] = "cluster_pvals",
    fig: matplotlib.figure.Figure | None = None,
    title: str | None = None,
    x_label: str = "Time [s]",
    y_label: str = "Frequency [Hz]",
    cbar_label: str = "Power [AU]",
    cbar_borderval: str | int | float = "auto",
    plot_pvals: bool = True,
    outpath: Path | str | None = None,
    show: bool = True,
    n_jobs: int = 1,
) -> matplotlib.figure.Figure:
    """Plot power, p-values and significant clusters."""

    if isinstance(power_b, (int, float)):
        power_av = power_a.mean(axis=0)
    else:
        power_av = power_b.mean(axis=0) - power_a.mean(axis=0)

    if isinstance(cbar_borderval, str):
        if cbar_borderval != "auto":
            raise ValueError(
                "`cbar_borderval` must be either an int, float or"
                f" 'auto'. Got: {cbar_borderval}."
            )
        cbar_borderval = min(power_av.max(), np.abs(power_av.min()))

    if not fig:
        if plot_pvals:
            ncols = 3
        else:
            ncols = 1
        fig, axs = plt.subplots(
            nrows=1,
            ncols=ncols,
            figsize=(8, 2.4),
            sharex=True,
            sharey=True,
        )
    else:
        axs = fig.axes
    if not isinstance(axs, np.ndarray):
        if not isinstance(axs, Sequence):
            axs = [axs]
        axs = np.array(axs)
    for ax in axs:
        ax.set_xlabel(x_label)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axs[0].set_ylabel(y_label)
    pos_0 = axs[0].imshow(
        power_av,
        extent=extent,
        cmap="viridis",
        aspect="auto",
        origin="lower",
        vmin=cbar_borderval * -1,
        vmax=cbar_borderval,
    )
    fig.colorbar(pos_0, ax=axs[0], label=cbar_label)

    if correction_method == "cluster_pvals":
        p_values, _, cluster_arr = pte_stats.cluster_analysis_2d_from_pvals(
            data_a=power_a,
            data_b=power_b,
            alpha=alpha,
            n_perm=n_perm,
            only_max_cluster=False,
            two_tailed=True,
            n_jobs=n_jobs,
        )
    elif correction_method == "cluster":
        p_values, _, cluster_arr = pte_stats.cluster_analysis_2d(
            data_a=power_a,
            data_b=power_b,
            alpha=alpha,
            n_perm=n_perm,
            only_max_cluster=False,
            two_tailed=True,
            n_jobs=n_jobs,
        )
    else:
        raise ValueError(
            f"Unknown correction method. Got: {correction_method}."
            "Must be one of 'cluster_pvals' or 'cluster'."
        )

    squared = np.zeros(power_av.shape[:], dtype=int)
    if cluster_arr:
        for cluster in cluster_arr:
            if cluster[0].size > 20:
                squared[cluster] = 1
        if squared.any():
            axs[0].contour(
                squared,
                levels=[0.99],
                extent=extent,
                origin="lower",
                colors="black",
            )
            if axs.size > 1:
                axs[1].contour(
                    squared,
                    levels=[0.99],
                    extent=extent,
                    origin="lower",
                    colors="black",
                )
    if axs.size > 1:
        pos_1 = axs[1].imshow(
            p_values,
            extent=extent,
            norm="log",
            cmap="viridis_r",
            aspect="auto",
            origin="lower",
        )
        fig.colorbar(pos_1, ax=axs[1], label="P [log]")

    if axs.size > 2:
        pos_2 = axs[2].imshow(
            squared,
            extent=extent,
            cmap="binary",
            aspect="auto",
            origin="lower",
            vmin=0,
            vmax=1,
        )
        cbar = fig.colorbar(
            pos_2, ax=axs[2], label=f"Signif. Clusters [P≤{alpha}]"
        )
        cbar.ax.set_yticks([0, 1])

    fig.suptitle(title)
    if outpath:
        fig.savefig(outpath)
    if show:
        plt.show()
    return fig

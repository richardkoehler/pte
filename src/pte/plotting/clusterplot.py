"""Module for plotting clusters."""
from pathlib import Path

import matplotlib.figure
import numpy as np
import pte_stats
from matplotlib import pyplot as plt


def clusterplot_combined(
    power_a: np.ndarray,
    power_b: np.ndarray | int | float,
    extent: tuple | list,
    alpha: float = 0.05,
    n_perm: int = 100,
    fig: matplotlib.figure.Figure | None = None,
    title: str | None = None,
    label_cbar: str = "Power [AU]",
    yscale: str = "log",
    borderval_cbar: str | int | float = "auto",
    outpath: Path | str | None = None,
    show: bool = True,
    n_jobs: int = 1,
) -> matplotlib.figure.Figure:
    """Plot power, p-values and significant clusters."""

    if isinstance(power_b, (int, float)):
        power_av = power_a.mean(axis=0)
    else:
        power_av = power_b.mean(axis=0) - power_a.mean(axis=0)

    if isinstance(borderval_cbar, str):
        if borderval_cbar != "auto":
            raise ValueError(
                "`border_val` must be either an int, float or"
                f" 'auto'. Got: {borderval_cbar}."
            )
        borderval_cbar = min(power_av.max(), np.abs(power_av.min()))

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
    axs[0].set_ylabel("Frequency [Hz]")
    axs[0].set_xlabel("Time [s]")
    # if yscale:  #  != "lin":
    #     for ax in axs:
    #         ax.set_yticks([2, 10, 100, 1000])
    #         # ax.semilogy()
    #         # ax.set_yscale("linear")
    # Plot averaged power
    pos_0 = axs[0].imshow(
        power_av,
        extent=extent,
        cmap="viridis",
        aspect="auto",
        origin="lower",
        vmin=borderval_cbar * -1,
        vmax=borderval_cbar,
    )
    fig.colorbar(pos_0, ax=axs[0], label=label_cbar)

    p_values, _, cluster_arr = pte_stats.cluster_analysis_2d(
        data_a=power_a,
        data_b=power_b,
        alpha=alpha,
        n_perm=n_perm,
        only_max_cluster=False,
        n_jobs=n_jobs,
    )
    # Get 2D clusters
    # p_values = pte_stats.permutation_2d(
    #     data_a=power_a,
    #     data_b=power_b,
    #     n_perm=n_perm,
    #     two_tailed=True,
    # )
    # cluster_arr, n_clusters = pte_stats.clusters_from_pvals(
    #     p_values,
    #     alpha=alpha,
    #     correction_method="fdr",
    #     n_perm=n_perm,
    #     min_cluster_size=1,
    # )

    # Plot p-values
    # p_values_masked = np.ma.masked_where(alpha < p_values, p_values)
    pos_1 = axs[1].imshow(
        p_values,
        extent=extent,
        norm="log",
        cmap="viridis_r",
        aspect="auto",
        origin="lower",
    )
    axs[1].contour(
        p_values,
        levels=[alpha],
        extent=extent,
        origin="lower",
        colors="black",
    )
    fig.colorbar(pos_1, ax=axs[1], label="P [log]")

    # Plot significant clusters
    squared = np.zeros(power_a.shape[1:])
    if cluster_arr:
        for cluster in cluster_arr:
            squared[cluster] = 1
    np.expand_dims(squared, axis=0)

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

    if np.any(squared):
        axs[0].contour(
            squared,
            levels=[0.9],
            extent=extent,
            origin="lower",
            colors="black",
        )

    fig.suptitle(title)
    fig.tight_layout()
    if outpath:
        fig.savefig(outpath)
    if show:
        plt.show()
    else:
        plt.close()
    return fig

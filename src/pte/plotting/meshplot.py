"""Module for plotting quantitave results onto brain structures."""

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io
from matplotlib import cm, collections, figure, ticker
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

RESOURCES = Path(__file__).parent / "resources"
# faces = scipy.io.loadmat(r"resources/faces.mat")
# grid = scipy.io.loadmat(r"resources/grid.mat")["grid"]


def meshplot_2d_compare(
    key: str,
    data_left: pd.DataFrame,
    data_right: pd.DataFrame,
    label_left: str = "",
    label_right: str = "",
    lims_left: tuple[tuple, tuple] = ((None, None), (None, None)),
    lims_right: tuple[tuple, tuple] = ((None, None), (None, None)),
    outpath: Path | str | None = None,
    dot_size: int = 20,
    ratio_cortex_subcortex: int | float = 4,
    title: str | None = None,
    invert_colors_left: bool = False,
    invert_colors_right: bool = False,
    show: bool = True,
    verbose: bool | str | int | None = True,
) -> figure.Figure:
    """Plot data on both hemispheres in a 2D view."""
    vertices = scipy.io.loadmat(str(RESOURCES / "vertices.mat"))
    stn_surf = scipy.io.loadmat(str(RESOURCES / "stn_surf.mat"))
    # x_ = stn_surf["vertices"][::2, 0]
    # y_ = stn_surf["vertices"][::2, 1]
    x_ecog = vertices["Vertices"][::1, 0]
    y_ecog = vertices["Vertices"][::1, 1]
    x_subcort = stn_surf["vertices"][::1, 0]
    y_subcort = stn_surf["vertices"][::1, 1]

    color = "black"
    height_subcort = 1
    height_cortex = ratio_cortex_subcortex * height_subcort
    fig, axes = plt.subplots(
        nrows=2,
        ncols=1,
        # figsize=(8, 9),
        frameon=False,
        gridspec_kw={"height_ratios": [height_cortex, height_subcort]},
    )
    axes = np.asarray(axes)

    for i, (x, y) in enumerate([(x_ecog, y_ecog), (x_subcort, y_subcort)]):
        axes[i].scatter(x, y, c="gray", s=0.001)
        axes[i].axes.set_aspect("equal", anchor="C")

    ecog_list, stn_list, cbar_list = [], [], []

    for data, side, lims, invert_colors in (
        (data_left, "left", lims_left, invert_colors_left),
        (data_right, "right", lims_right, invert_colors_right),
    ):
        plot_cort, plot_stn = _plot_single_hem(
            data=data,
            axes=axes,
            side=side,
            lims=lims,
            dot_size=dot_size,
            key=key,
            reverse_cmap=invert_colors,
            verbose=verbose,
        )
        ecog_list.append(plot_cort)
        stn_list.append(plot_stn)
        cbar_list.append(side)

    for ax in axes:
        ax.axes.xaxis.set_ticks([])
        ax.axes.yaxis.set_ticks([])
        ax.axis("off")

    label_cbars = [label_left, label_right]
    for idx, (plot_cort, plot_stn, location) in enumerate(
        zip(ecog_list, stn_list, cbar_list, strict=True)
    ):
        for i, (pos, ratio) in enumerate(
            ((plot_cort, ratio_cortex_subcortex), (plot_stn, 1))
        ):
            divider = make_axes_locatable(axes[i])
            cax = divider.append_axes(position=location, size="3%", pad=0.05)
            aspect = 6 * ratio
            cax.axis("off")
            cbar = fig.colorbar(
                pos,
                ax=cax,
                location=location,
                aspect=aspect,
                pad=0.0,
                shrink=0.8,
                ticklocation=location,
            )
            cbar.set_label(label_cbars[idx], color=color)
            cbar.ax.yaxis.set_major_locator(ticker.MaxNLocator(5))
            ticks_loc = cbar.ax.get_yticks().tolist()
            cbar.ax.yaxis.set_major_locator(ticker.FixedLocator(ticks_loc))
            cbar.ax.set_yticklabels(
                labels=np.round(cbar.get_ticks(), 2), color=color
            )
            cbar.outline.set_edgecolor(color)  # type: ignore[operator]
    if title is not None:
        fig.suptitle(title)
    if outpath is not None:
        fig.savefig(outpath, bbox_inches="tight")
    if show:
        plt.show(block=True)
    return fig


def _get_lims(data: np.ndarray, num_devs: int | float) -> tuple[float, float]:
    """Get lower and upper limit in standard deviations."""
    mean = np.nanmean(data)
    std = np.nanstd(data)
    upper = mean + num_devs * std
    lower = mean - num_devs * std
    return lower, upper


def _plot_single_hem(
    data: pd.DataFrame,
    axes: np.ndarray,
    side: str,
    lims: tuple,
    dot_size: int,
    key: str,
    reverse_cmap: bool,
    verbose: bool | str | int | None,
) -> tuple[collections.PathCollection, collections.PathCollection]:
    """Plot data for a single hemisphere."""
    data = data.dropna(axis=0, subset=["x", "y", "z"])
    factor = -1.0 if side == "left" else 1.0
    cmap = copy.copy(cm.get_cmap("viridis"))
    if reverse_cmap:
        cmap = cmap.reversed()
    cmap.set_bad(color="magenta")

    pos_list = []
    for i, ch_type in enumerate(("ECOG", "LFP")):
        ind = data["name"].str.contains(ch_type)
        coord = data.loc[ind, ["x", "y"]].to_numpy(dtype=np.float32)
        values = data.loc[ind, key].to_numpy(dtype=np.float32)
        lims_list = list(lims[i])
        for idx, lim in enumerate(lims_list):
            if lim:
                new_lims = np.round(_get_lims(data=values, num_devs=lim), 2)
                lims_list[idx] = new_lims[idx]
        if verbose:
            print(f"Using color limits: {lims_list}.")
        pos = axes[i].scatter(
            np.abs(coord[:, 0]) * factor,
            coord[:, 1],
            c=values,
            s=dot_size,
            alpha=0.8,
            cmap=cmap,
            vmin=lims_list[0],
            vmax=lims_list[1],
            plotnonfinite=True,
        )
        pos_list.append(pos)
    return pos_list[0], pos_list[1]

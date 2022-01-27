"""Module for plotting decoding results."""
import os
from itertools import combinations, product
from pathlib import Path
from typing import Callable, Iterable, Optional, Union

import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
from matplotlib import axes, cm, collections, figure, patheffects
from matplotlib import pyplot as plt
from statannotations import Annotator
from statannotations.stats import StatTest

import pte


def violinplot_results(
    data: pd.DataFrame,
    outpath: Union[str, Path],
    x: str,
    y: str,
    hue: Optional[str] = None,
    order: Optional[Iterable] = None,
    hue_order: Optional[Iterable] = None,
    stat_test: Optional[Union[Callable, str]] = "Permutation",
    alpha: Optional[float] = 0.05,
    add_lines: Optional[str] = None,
    title: Optional[str] = "Classification Performance",
    figsize: Union[tuple, str] = "auto",
) -> None:
    """Plot performance as combined boxplot and stripplot."""
    if not order:
        order = data[x].unique()

    if hue and not hue_order:
        hue_order = data[hue].unique()

    if figsize == "auto":
        hue_factor = 1 if not hue else len(hue_order)
        figsize = (1.1 * len(order) * hue_factor, 4)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)

    ax = sns.violinplot(
        x=x,
        y=y,
        hue=hue,
        order=order,
        hue_order=hue_order,
        data=data,
        palette="viridis",
        inner="box",
        width=0.9,
        alpha=0.8,
        ax=ax,
    )

    ax = sns.swarmplot(
        x=x,
        y=y,
        hue=hue,
        order=order,
        hue_order=hue_order,
        data=data,
        color="white",
        alpha=0.5,
        dodge=True,
        s=6,
        ax=ax,
    )

    if stat_test:
        _add_stats(
            ax=ax,
            data=data,
            x=x,
            y=y,
            order=order,
            hue=hue,
            hue_order=hue_order,
            stat_test=stat_test,
            alpha=alpha,
            location="outside",
        )

    if hue:
        handles, labels = ax.get_legend_handles_labels()
        new_labels = [
            label.replace(" ", "\n") for label in labels[: len(labels) // 2]
        ]
        _ = plt.legend(
            handles[: len(handles) // 2],
            new_labels,
            bbox_to_anchor=(1.02, 1),
            loc=2,
            borderaxespad=0.0,
            title=hue,
            labelspacing=0.7,
        )

    xlabels = [item.get_text() for item in ax.get_xticklabels()]
    new_xlabels = [xtick.replace(" ", "\n") for xtick in xlabels]
    ax.set_xticklabels(new_xlabels)
    ax.set_title(title, fontsize="medium", y=1.02)

    if add_lines:
        _add_lines(
            ax=ax, data=data, x=x, y=y, order=order, add_lines=add_lines
        )

    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight", dpi=450)
    plt.show(block=True)


def boxplot_results(
    data: pd.DataFrame,
    outpath: Union[str, Path],
    x: str,
    y: str,
    hue: Optional[str] = None,
    order: Optional[Iterable] = None,
    hue_order: Optional[Iterable] = None,
    stat_test: Optional[Union[Callable, str]] = "Permutation",
    alpha: Optional[float] = 0.05,
    add_lines: Optional[str] = None,
    add_median_labels: bool = False,
    title: Optional[str] = "Classification Performance",
    figsize: Union[tuple, str] = "auto",
) -> None:
    """Plot performance as combined boxplot and stripplot."""
    # data = data[["Channels", "Balanced Accuracy", "Subject"]]

    color = "black"
    alpha_box = 0.5

    if not order:
        order = data[x].unique()

    if hue and not hue_order:
        hue_order = data[hue].unique()

    if figsize == "auto":
        hue_factor = 1 if not hue else len(hue_order)
        figsize = (1.1 * len(order) * hue_factor, 4)

    plt.figure(figsize=figsize)

    ax = sns.boxplot(
        x=x,
        y=y,
        hue=hue,
        order=order,
        hue_order=hue_order,
        data=data,
        palette="viridis",
        boxprops=dict(alpha=alpha_box),
        showcaps=True,
        showbox=True,
        showfliers=False,
        notch=False,
        width=0.9,
        whiskerprops={
            "linewidth": 2,
            "zorder": 10,
            "alpha": alpha_box,
            "color": color,
        },
        capprops={"alpha": alpha_box, "color": color},
        medianprops=dict(
            linestyle="-", linewidth=5, color=color, alpha=alpha_box
        ),
    )

    if add_median_labels:
        _add_median_labels(ax)

    sns.swarmplot(
        x=x,
        y=y,
        hue=hue,
        order=order,
        hue_order=hue_order,
        data=data,
        palette="viridis",
        dodge=True,
        s=6,
        ax=ax,
    )

    if hue:
        handles, labels = ax.get_legend_handles_labels()
        new_labels = [
            label.replace(" ", "\n") for label in labels[: len(labels) // 2]
        ]
        _ = plt.legend(
            handles[: len(handles) // 2],
            new_labels,
            bbox_to_anchor=(1.02, 1),
            loc=2,
            borderaxespad=0.0,
            title=hue,
            labelspacing=0.7,
        )

    xlabels = [item.get_text() for item in ax.get_xticklabels()]
    new_xlabels = [xtick.replace(" ", "\n") for xtick in xlabels]
    ax.set_xticklabels(new_xlabels)
    ax.set_title(title, fontsize="medium", y=1.02)

    if add_lines:
        _add_lines(
            ax=ax, data=data, x=x, y=y, order=order, add_lines=add_lines
        )

    if stat_test:
        _add_stats(
            ax,
            data,
            x,
            y,
            order,
            hue,
            hue_order,
            stat_test=stat_test,
            alpha=alpha,
        )

    plt.tight_layout()
    plt.savefig(outpath, bbox_inches="tight", dpi=450)
    plt.show(block=True)


def _add_lines(
    ax: axes.Axes,
    data: pd.DataFrame,
    x: str,
    y: str,
    order: list,
    add_lines: str,
):
    """Add lines connecting single dots"""
    data = data.sort_values(
        by=x, key=lambda k: k.map({item: i for i, item in enumerate(order)})
    )
    lines = (
        [[i, n] for i, n in enumerate(group)]
        for _, group in data.groupby([add_lines], sort=False)[y]
    )
    ax.add_collection(
        collections.LineCollection(lines, colors="grey", linewidths=1)
    )


def _add_stats(
    ax: axes.Axes,
    data: pd.DataFrame,
    x: str,
    y: str,
    order: Iterable,
    hue: Optional[str],
    hue_order: Optional[Iterable],
    stat_test: Union[str, StatTest.StatTest],
    alpha: float,
    location: str = "inside",
):
    """Perform statistical test and annotate graph."""
    if not hue:
        pairs = list(combinations(order, 2))
    else:
        pairs = [
            list(combinations(list(product([item], hue_order)), 2))
            for item in order
        ]
        pairs = [item for sublist in pairs for item in sublist]

    if stat_test == "Permutation":
        stat_test = StatTest.StatTest(
            func=_permutation_wrapper,
            n_perm=10000,
            alpha=alpha,
            test_long_name="Permutation Test",
            test_short_name="Perm.",
            stat_name="Effect Size",
        )
    annotator = Annotator.Annotator(
        ax=ax,
        pairs=pairs,
        data=data,
        x=x,
        y=y,
        hue=hue,
        hue_order=hue_order,
        order=order,
    )
    annotator.configure(
        alpha=alpha,
        test=stat_test,
        text_format="simple",
        loc=location,
        color="grey",
    )
    annotator.apply_and_annotate()


def lineplot_prediction(
    x: np.ndarray,
    y: np.ndarray,
    subpl_titles: Iterable,
    sfreq: Union[int, float],
    x_lims: tuple,
    outpath: Optional[Union[Path, str]] = None,
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    label: Optional[str] = "Distance from Hyperplane",
    y_label: str = None,
    threshold: Union[
        int, float, tuple[Union[int, float], Union[int, float]]
    ] = (0.0, 1.0),
    p_lim: float = 0.05,
    n_perm: int = 1000,
    correction_method: str = "cluster",
    two_tailed: bool = False,
    ylims=None,
    compare_xy: bool = False,
) -> figure.Figure:
    """Plot averaged time-locked predictions including statistical tests."""
    viridis = cm.get_cmap("viridis", 8)
    colors = viridis(4), viridis(2)

    nrows = 3 if compare_xy else 2

    fig, axs = plt.subplots(
        ncols=1, nrows=nrows, figsize=(6, 2.0 * nrows), sharex=False
    )

    n_samples = x.shape[0]

    for i, data in enumerate((x, y)):
        _single_lineplot(
            data=data,
            ax=axs[i],
            threshold=threshold,
            sfreq=sfreq,
            x_lims=x_lims,
            color=colors[i],
            subpl_title=subpl_titles[i],
            label=label,
            p_lim=p_lim,
            n_perm=n_perm,
            correction_method=correction_method,
        )

    if compare_xy:
        for i, data in enumerate([x, y]):
            label_ = (
                " - ".join((subpl_titles[i], label))
                if label
                else subpl_titles[i]
            )
            axs[2].plot(
                data.mean(axis=1),
                color=colors[i],
                label=label_,
            )
            axs[2].fill_between(
                np.arange(data.shape[0]),
                data.mean(axis=1) - stats.sem(data, axis=1),
                data.mean(axis=1) + stats.sem(data, axis=1),
                alpha=0.5,
                color=colors[i],
            )
        axs[2].set_title(
            f"{subpl_titles[0]} vs. {subpl_titles[1]}", fontsize="medium"
        )
        axs[2].set_xlabel("Time [s]")

        p_vals = pte.stats.timeseries_pvals(
            x=x, y=y, n_perm=n_perm, two_tailed=two_tailed
        )

        _pval_correction_lineplot(
            ax=axs[2],
            x=x,
            y=y,
            x_lims=x_lims,
            p_vals=p_vals,
            alpha=p_lim,
            n_perm=n_perm,
            correction_method=correction_method,
        )

    for ax in axs:
        ax.legend(loc="upper left", fontsize="small")
        if ylims:
            ax.set_ylim(ylims[0], ylims[1])
        xticks = np.arange(0, n_samples, sfreq)
        ax.set_xticks(xticks)
        ax.set_xticklabels(np.linspace(x_lims[0], x_lims[1], len(xticks)))
        ax.set_ylabel(y_label)
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
    if outpath:
        fig.savefig(os.path.normpath(outpath), bbox_inches="tight", dpi=300)
    plt.show(block=True)
    return fig


def _permutation_wrapper(x, y, n_perm) -> tuple:
    """Wrapper for statannotations to convert pandas series to numpy array."""
    if isinstance(x, pd.Series):
        x = x.values
    if isinstance(y, pd.Series):
        y = y.values
    return pte.stats.permutation_twosample(x=x, y=y, n_perm=n_perm)


def _pval_correction_lineplot(
    ax: axes.Axes,
    x: np.ndarray,
    y: np.ndarray,
    x_lims: tuple,
    p_vals: Iterable,
    alpha: float,
    correction_method: str,
    n_perm: Optional[int] = None,
) -> None:
    """Perform p-value correction for singe lineplot."""
    viridis = cm.get_cmap("viridis", 8)
    if y.ndim == 1:
        y = np.expand_dims(y, axis=1)

    clusters, cluster_count = pte.stats.clusters_from_pvals(
        p_vals=p_vals,
        alpha=alpha,
        correction_method=correction_method,
        n_perm=n_perm,
    )

    if cluster_count > 0:
        label = f"p-value <= {alpha}"
        x_labels = np.linspace(x_lims[0], x_lims[1], len(p_vals)).round(2)
        for cluster_idx in range(1, cluster_count + 1):
            index = np.where(clusters == cluster_idx)[0]
            # time_point = x_labels[index]
            lims = np.arange(index[0], index[-1] + 1)
            y_lims = y.mean(axis=1)[lims]
            if y_lims.size > 0:
                ax.fill_between(
                    x=lims,
                    y1=x.mean(axis=1)[lims],
                    y2=y_lims,
                    alpha=0.5,
                    color=viridis(7),
                    label=label,
                )
                label = None  # Avoid printing label multiple times
                # label_lims = lims[where]
                for i in [0, -1]:
                    ax.annotate(
                        str(x_labels[lims[i]]) + "s",
                        (lims[i], y_lims[i]),
                        xytext=(0.0, 15),
                        textcoords="offset points",
                        verticalalignment="center",
                        horizontalalignment="center",
                        arrowprops=dict(facecolor="black", arrowstyle="-"),
                    )


def _single_lineplot(
    data: np.ndarray,
    ax: axes.Axes,
    threshold: Union[Iterable, int, float],
    sfreq: Union[int, float],
    x_lims: tuple,
    color: tuple,
    label: Optional[str],
    subpl_title: str,
    p_lim: float,
    n_perm: int,
    correction_method: str,
) -> None:
    """Plot prediction line for single model."""
    (
        threshold_value,
        threshold_arr,
    ) = pte.decoding.timepoint.transform_threshold(
        threshold=threshold, sfreq=sfreq, data=data
    )

    # x = np.arange(data.shape[0])
    # lines = collections.LineCollection(
    #     [np.column_stack([x, dat_]) for dat_ in data.T],
    #     color=color,
    #     linewidth=1,
    #     alpha=0.5,
    # )
    # ax.add_collection(lines)

    ax.plot(data.mean(axis=1), color=color, label=label)
    ax.fill_between(
        np.arange(data.shape[0]),
        data.mean(axis=1) - stats.sem(data, axis=1),
        data.mean(axis=1) + stats.sem(data, axis=1),
        alpha=0.5,
        color=color,
        label=None,
    )
    ax.plot(
        threshold_arr,
        color="r",
        label="Threshold",
        alpha=0.5,
        linestyle="dashed",
    )

    p_vals = pte.stats.timeseries_pvals(
        x=data, y=threshold_value, n_perm=n_perm, two_tailed=False
    )

    _pval_correction_lineplot(
        ax=ax,
        x=data,
        y=threshold_arr,
        x_lims=x_lims,
        p_vals=p_vals,
        alpha=p_lim,
        n_perm=n_perm,
        correction_method=correction_method,
    )

    ax.set_title(subpl_title, fontsize="medium")


def _add_median_labels(ax: axes.Axes, add_borders: bool = False) -> None:
    """Add median labels to boxplot."""
    lines = ax.get_lines()
    # determine number of lines per box (this varies with/without fliers)
    boxes = [c for c in ax.get_children() if type(c).__name__ == "PathPatch"]
    lines_per_box = int(len(lines) / len(boxes))
    # iterate over median lines
    for median in lines[4 : len(lines) : lines_per_box]:
        # display median value at center of median line
        x, y = (data.mean() for data in median.get_data())
        # choose value depending on horizontal or vertical plot orientation
        value = (
            x if (median.get_xdata()[1] - median.get_xdata()[0]) == 0 else y
        )
        text = ax.text(
            x=x,
            y=y,
            s=f"{value:.3f}",
            ha="center",
            va="center",
            fontweight="light",
            color="white",
            fontsize="medium",
        )
        if add_borders:
            # create median-colored border around white text for contrast
            text.set_path_effects(
                [
                    patheffects.Stroke(
                        linewidth=0.0, foreground=median.get_color()
                    ),
                    patheffects.Normal(),
                ]
            )

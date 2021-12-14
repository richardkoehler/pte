"""Module for plotting decoding results."""
from itertools import combinations, product
import os
from pathlib import Path
from typing import Callable, Iterable, Optional, Union

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import axes, cm, collections, patheffects
from matplotlib import pyplot as plt
from statannotations.Annotator import Annotator
from statannotations.stats.StatTest import StatTest

from statsmodels.stats.multitest import fdrcorrection

import pte


def boxplot_performance(
    data: pd.DataFrame,
    outpath: Union[str, Path],
    x: str,
    y: str,
    hue: Optional[str] = None,
    order: Optional[Iterable] = None,
    hue_order: Optional[Iterable] = None,
    stat_test: Optional[str] = "Permutation",
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
        figsize = (1 * len(order) * hue_factor, 4)

    plt.figure(figsize=figsize)

    ax = sns.boxplot(
        x=x,
        y=y,
        hue=hue,
        order=order,
        hue_order=hue_order,
        data=data,
        palette="viridis",
        # showmeans=True,
        meanline=False,
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

    sns.stripplot(
        x=x,
        y=y,
        hue=hue,
        order=order,
        hue_order=hue_order,
        data=data,
        palette="viridis",
        dodge=True,
        jitter=1.0,
        s=7,
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
        _add_stats(ax, data, x, y, order, hue, hue_order, stat_test=stat_test)

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
    lc = collections.LineCollection(lines, colors="grey", linewidths=1)
    ax.add_collection(lc)


def _add_stats(
    ax: axes.Axes,
    data: pd.DataFrame,
    x: str,
    y: str,
    order: Iterable,
    hue: Optional[str],
    hue_order: Optional[Iterable],
    stat_test: str,
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
        stat_test = StatTest(
            func=_permutation_wrapper,
            n_perm=10000,
            test_long_name="Permutation Test",
            test_short_name="Perm.",
            stat_name="Effect Size",
        )
    annotator = Annotator(
        ax,
        pairs,
        data=data,
        x=x,
        y=y,
        hue=hue,
        hue_order=hue_order,
        order=order,
    )
    annotator.configure(
        test=stat_test, text_format="star", loc="inside", color="grey"
    )
    annotator.apply_and_annotate()


def lineplot_prediction(
    x: np.ndarray,
    y: np.ndarray,
    subpl_titles: Iterable,
    sfreq: Union[int, float],
    x_lims: tuple,
    fpath: Optional[Union[Path, str]] = None,
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    label: str = "Distance from Hyperplane",
    y_label: str = None,
    threshold: Union[int, float, tuple] = (0.0, 1.0),
    p_lim: float = 0.05,
    n_perm: int = 1000,
    correction_method: str = "cluster",
    two_tailed: bool = False,
    ylims=None,
    compare_xy: bool = False,
) -> None:
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
                data.mean(axis=1), color=colors[i], label=label_,
            )
            axs[2].fill_between(
                np.arange(data.shape[0]),
                data.mean(axis=1) - data.std(axis=1),
                data.mean(axis=1) + data.std(axis=1),
                alpha=0.5,
                color=colors[i],
            )
        axs[2].set_title(
            f"{subpl_titles[0]} vs. {subpl_titles[1]}", fontsize="medium"
        )
        axs[2].set_xlabel("Time [s]")

        p_vals = _get_p_vals(x=x, y=y, n_perm=n_perm, two_tailed=two_tailed)

        _pval_correction_lineplot(
            ax=axs[2],
            x=x,
            y=y,
            x_lims=x_lims,
            p_vals=p_vals,
            p_lim=p_lim,
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
    if fpath:
        fig.savefig(os.path.normpath(fpath), bbox_inches="tight", dpi=300)
    plt.show(block=True)


def _permutation_wrapper(x, y, n_perm) -> Callable:
    """Wrapper for statannotations to convert pandas series to numpy array."""
    if isinstance(x, pd.Series):
        x = x.values
    if isinstance(y, pd.Series):
        y = y.values
    return pte.stats.permutation_twosample(x, y, n_perm)


def _add_median_labels(ax: axes.Axes, add_borders: bool = False) -> None:
    """"""
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
        if add_borders:
            # create median-colored border around white text for contrast
            text = ax.text(
                x,
                y,
                f"{value:.3f}",
                ha="center",
                va="center",
                fontweight="ultralight",
                color="white",
                fontsize="medium",
            )
            text.set_path_effects(
                [
                    patheffects.Stroke(
                        linewidth=0.0, foreground=median.get_color()
                    ),
                    patheffects.Normal(),
                ]
            )


def _pval_correction_lineplot(
    ax: axes.Axes,
    x: np.ndarray,
    y: np.ndarray,
    x_lims: tuple,
    p_vals: Iterable,
    p_lim: float,
    correction_method: str,
    n_perm: Optional[int] = None,
) -> None:
    """Perform p-value correction for singe lineplot."""
    viridis = cm.get_cmap("viridis", 8)
    if y.ndim == 1:
        y = np.expand_dims(y, axis=1)

    signif = np.where(p_vals <= p_lim)[0]
    if signif.size > 0:
        if correction_method == "cluster":
            _, signif = pte.stats.clusterwise_pval_numba(
                p_arr=p_vals, p_sig=p_lim, n_perm=n_perm
            )
        elif correction_method == "fdr":
            rejected, _ = fdrcorrection(
                pvals=p_vals, alpha=p_lim, method="poscorr", is_sorted=False
            )
            signif = [np.where(rejected)[0]]
        else:
            raise ValueError(
                f"`correction_method` must be one of either `cluster` or `fdr`. \
                Got:{correction_method}."
            )
        if signif:
            x_labels = np.linspace(x_lims[0], x_lims[1], len(x)).round(2)
            label = f"p-value <= {p_lim}"
            for sig in signif:
                lims = np.arange(sig[0], sig[-1] + 1)
                y_lims = y.mean(axis=1)[lims]
                ax.fill_between(
                    lims,
                    x.mean(axis=1)[lims],
                    y_lims,
                    alpha=0.5,
                    color=viridis(7),
                    label=label,
                )
                label = None  # Avoid printing label multiple times
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
    label: str,
    subpl_title: str,
    p_lim: float,
    n_perm: int,
    correction_method: str,
) -> None:
    """Plot prediction line for single model."""
    threshold_value, threshold_arr = _transform_threshold(
        threshold=threshold, sfreq=sfreq, data=data
    )

    p_vals = _get_p_vals(
        x=data, y=threshold_value, n_perm=n_perm, two_tailed=False
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
        label=None,
    )

    _pval_correction_lineplot(
        ax=ax,
        x=data,
        y=threshold_arr,
        x_lims=x_lims,
        p_vals=p_vals,
        p_lim=p_lim,
        n_perm=n_perm,
        correction_method=correction_method,
    )

    ax.set_title(subpl_title, fontsize="medium")


def _get_p_vals(
    x: Iterable, y: Union[Iterable, int, float], n_perm: int, two_tailed: bool
):
    """Calculate sample-wise p-values."""
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


def _transform_threshold(
    threshold: Union[Iterable, int, float],
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
    threshold_arr = np.ones_like(data[:, 0]) * threshold_value
    return threshold_value, threshold_arr


def _add_median_labels(ax: axes.Axes) -> None:
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
        _ = ax.text(
            x=x,
            y=y,
            s=f"{value:.3f}",
            ha="center",
            va="center",
            fontweight="light",
            color="white",
            fontsize="medium",
        )

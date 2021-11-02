"""This module needs to be refactored and will be removed in the future"""

import math
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import zscore


def plot_features(
    features,
    events,
    ch_names,
    path,
    sfreq,
    time_begin,
    time_end,
    dist_onset,
    dist_end,
):
    """"""
    dist_onset = int(dist_onset * sfreq)
    dist_end = int(dist_end * sfreq)
    samp_begin = int(time_begin * sfreq)
    samp_end = int(time_end * sfreq + 1)
    x = []
    data = features.values
    for i, ind in enumerate(np.arange(0, len(events), 2)):
        append = True
        if i == 0:
            data_plot = data[events[ind] + samp_begin : events[ind] + samp_end]
            if data_plot.shape[0] != samp_end - samp_begin:
                append = False
        elif (events[ind] - dist_onset) - (events[ind - 1] + dist_end) <= 0:
            append = False
        else:
            data_plot = data[events[ind] + samp_begin : events[ind] + samp_end]
            if data_plot.shape[0] != samp_end - samp_begin:
                append = False
        if append:
            x.extend(np.expand_dims(data_plot, axis=0))
    x = np.mean(np.stack(x, axis=0), axis=0)
    features = pd.DataFrame(x, columns=features.columns)

    n_rows = 2
    n_cols = int(math.ceil(len(ch_names) / n_rows))
    fig, axs = plt.subplots(
        figsize=(n_cols * 3, 5),
        nrows=n_rows,
        ncols=n_cols,
        dpi=300,
        sharex=False,
        sharey=True,
    )
    ind = 0
    ch_names.sort()
    for row in np.arange(n_rows):
        for col in np.arange(n_cols):
            if ind < len(ch_names):
                ch_name = ch_names[ind]
                cols = [col for col in features.columns if ch_name in col]
                # cols = [col for col in features.columns
                #       if ch_name in col and "diff" not in col]
                yticks = [
                    "Theta",
                    "Alpha",
                    "Low Beta",
                    "High Beta",
                    "Low Gamma",
                    "High Gamma",
                    "High Frequency Activity",
                    "Theta Derivation",
                    "Alpha Derivation",
                    "Low Beta Derivation",
                    "High Beta Derivation",
                    "Low Gamma Derivation",
                    "High Gamma Derivation",
                    "High Frequency Activity Derivation",
                ]
                # yticks = ["Theta", "Alpha", "Low Beta", "High Beta",
                #         "Low Gamma", "High Gamma", "High Frequency Activity"]
                x = features[cols].values
                ax = axs[row, col]
                ax.imshow(
                    zscore(x, axis=0).T,
                    cmap="viridis",
                    aspect="auto",
                    origin="lower",
                    vmin=-3.0,
                    vmax=3.0,
                )
                ax.set_yticks(np.arange(len(cols)))
                ax.set_yticklabels(yticks)
                ax.set_xticks(np.arange(0, x.shape[0] + 1, sfreq))
                ax.set_xticklabels(
                    np.arange(time_begin, time_end + 1, 1, dtype=int)
                )
                ax.set_title(str(ch_name))
                ind += 1
    for ax in axs.flat:
        ax.set(xlabel="Time [s]", ylabel="Features")
    fig.suptitle("Movement Aligned Features - Individual Channels")
    if not os.path.isdir(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)

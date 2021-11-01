"""This module needs to be refactored and will be removed in the future"""

import math
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import zscore


def generate_outpath(
    root,
    feature_file,
    classifier,
    target_beg,
    target_en,
    use_channels_,
    optimize,
    use_times,
):
    """"""
    clf_str = "_" + classifier + "_"

    target_str = (
        "movement_"
        if target_en == "MovementEnd"
        else "mot_intention_" + str(target_beg) + "_" + str(target_en) + "_"
    )
    ch_str = use_channels_ + "_chs_"
    opt_str = "opt_" if optimize else "no_opt_"
    out_name = (
        feature_file
        + clf_str
        + target_str
        + ch_str
        + opt_str
        + str(use_times * 100)
        + "ms"
    )
    return os.path.join(root, feature_file, out_name)


def events_from_label(label_data, verbose=False):
    """

    Parameters
    ----------
    label_data
    verbose

    Returns
    -------

    """
    label_diff = np.zeros_like(label_data, dtype=int)
    label_diff[1:] = np.diff(label_data)
    events_ = np.nonzero(label_diff)[0]
    if verbose:
        print(f"Number of events detected: {len(events_) / 2}")
    return events_


def get_target_df(targets, features_df):
    """"""
    i = 0
    target_df = pd.DataFrame()
    while len(target_df.columns) == 0:
        target_pick = targets[i]
        col_picks = [
            col for col in features_df.columns if target_pick in col.lower()
        ]
        for col in col_picks[:1]:
            target_df[col] = features_df[col]
        i += 1
    if len(col_picks[:1]) > 1:
        raise ValueError(f"Multiple targets found: {col_picks}")
    print("Target channel used: ", target_df.columns)
    return target_df


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


def get_feature_df(features, use_features, use_times):
    """

    Parameters
    ----------
    features
    use_features
    use_times

    Returns
    -------

    """
    # Extract features to use from dataframe
    column_picks = [
        col
        for col in features.columns
        if any([pick in col for pick in use_features])
    ]
    used_features = features[column_picks]

    # Initialize list of features to use
    feat_list = [
        used_features.rename(
            columns={col: col + "_100_ms" for col in used_features.columns}
        )
    ]

    # Use additional features from previous time points
    # use_times = 1 means no features from previous time points are
    # being used
    for s in np.arange(1, use_times):
        feat_list.append(
            used_features.shift(s, axis=0).rename(
                columns={
                    col: col + "_" + str((s + 1) * 100) + "_ms"
                    for col in used_features.columns
                }
            )
        )

    # Return final features dataframe
    return pd.concat(feat_list, axis=1).fillna(0.0)

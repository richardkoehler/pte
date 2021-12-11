"""Module for running decoding experiments."""
import os
import sys
from pathlib import Path
from typing import Iterable, Union

import numpy as np
import pandas as pd

from ..settings import PATH_PYNEUROMODULATION
from .decode import get_decoder
from .run import Runner

sys.path.insert(0, PATH_PYNEUROMODULATION)

from pyneuromodulation.nm_reader import NM_Reader


def run_experiment(
    features_root,
    feature_file,
    classifier,
    label_channels,
    target_begin,
    target_end,
    optimize,
    balancing,
    out_root,
    use_channels,
    use_features,
    cross_validation,
    scoring="balanced_accuracy",
    feature_importance=False,
    plot_target_channels=None,
    artifact_channels=None,
    bad_events_path=None,
    pred_mode="classify",
    use_times=1,
    dist_onset=2.0,
    dist_end=0.5,
    excep_dist_end=0.5,
    exceptions=None,
    save_plot=True,
    show_plot=False,
    verbose=True,
) -> None:
    """Run prediction experiment."""

    if verbose:
        print("Using file: ", feature_file)

    nm_reader = NM_Reader(feature_path=features_root)
    features = nm_reader.read_features(feature_file)
    settings = nm_reader.read_settings(feature_file)

    # Pick label for classification
    label_df = None
    for label_channel in label_channels:
        if label_channel in features.columns:
            label_df = nm_reader.read_label(label_channel)
            break
    if label_df is None:
        print(
            f"No valid label found. Labels given: {label_channels}. Discarding file: {feature_file}"
        )
        return

    bad_events = None
    if bad_events_path:
        bad_events_path = Path(bad_events_path)
        if bad_events_path.is_dir():
            basename = Path(feature_file).stem
            bad_events_path = bad_events_path / (basename + "_bad_epochs.csv")
        if not bad_events_path.exists():
            print(f"No bad epochs file found for: {str(feature_file)}")
        else:
            bad_events = pd.read_csv(
                bad_events_path, index_col=0
            ).event_id.values

    # Pick target for plotting predictions
    target_df = _get_target_df(plot_target_channels, features)

    features_df = _get_feature_df(features, use_features, use_times)

    # Pick artifact channel
    artifacts = None
    if artifact_channels:
        artifacts = _get_target_df(artifact_channels, features).values

    # Generate output file name
    out_path = _generate_outpath(
        out_root,
        feature_file,
        classifier,
        target_begin,
        target_end,
        use_channels,
        optimize,
        use_times,
    )

    decoder = get_decoder(
        classifier=classifier,
        scoring=scoring,
        balancing=balancing,
        optimize=optimize,
    )

    # Initialize Runner instance
    runner = Runner(
        features=features_df,
        target_df=target_df,
        label_df=label_df,
        artifacts=artifacts,
        bad_events=bad_events,
        ch_names=settings["ch_names"],
        sfreq=settings["sampling_rate_features"],
        classifier=classifier,
        balancing=balancing,
        optimize=optimize,
        decoder=decoder,
        target_begin=target_begin,
        target_end=target_end,
        dist_onset=dist_onset,
        dist_end=dist_end,
        exception_files=exceptions,
        excep_dist_end=excep_dist_end,
        use_channels=use_channels,
        pred_begin=-3.0,
        pred_end=2.0,
        pred_mode=pred_mode,
        cv_outer=cross_validation,
        feature_importance=feature_importance,
        show_plot=show_plot,
        save_plot=save_plot,
        out_file=out_path,
        verbose=verbose,
    )
    runner.run()


def _generate_outpath(
    root: str,
    feature_file: str,
    classifier: str,
    target_begin: Union[str, int, float],
    target_end: Union[str, int, float],
    use_channels: str,
    optimize: bool,
    use_times: int,
) -> str:
    """Generate file name for output files."""
    if target_begin == 0.0:
        target_begin = "trial_begin"
    if target_end == 0.0:
        target_end = "trial_begin"
    target_str = "_".join(("decode", str(target_begin), str(target_end)))
    clf_str = "_".join(("model", classifier))
    ch_str = "_".join(("chs", use_channels))
    opt_str = "opt_yes" if optimize else "opt_no"
    feat_str = "_".join(("feats", str(use_times * 100), "ms"))
    out_name = "_".join((target_str, clf_str, ch_str, opt_str, feat_str))
    return os.path.join(root, out_name, feature_file, feature_file)


def _get_feature_df(
    features: pd.DataFrame, use_features: Iterable, use_times: int
) -> pd.DataFrame:
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
    for use_time in np.arange(1, use_times):
        feat_list.append(
            used_features.shift(use_time, axis=0).rename(
                columns={
                    col: col + "_" + str((use_time + 1) * 100) + "_ms"
                    for col in used_features.columns
                }
            )
        )

    # Return final features dataframe
    return pd.concat(feat_list, axis=1).fillna(0.0)


def _events_from_label(
    label_data: np.ndarray, verbose: bool = False
) -> np.ndarray:
    """

    Parameters
    ----------
    label_data
    verbose

    Returns
    -------
    events
    """
    label_diff = np.zeros_like(label_data, dtype=int)
    label_diff[1:] = np.diff(label_data)
    if label_data[0] != 0:
        label_diff[0] = 1
    if label_data[-1] != 0:
        label_diff[-1] = -1
    events = np.nonzero(label_diff)[0]
    if verbose:
        print(f"Number of events detected: {len(events) / 2}")
    return events


def _get_target_df(
    targets: Iterable, features_df: pd.DataFrame, verbose: bool = False
) -> pd.DataFrame:
    """Extract target DataFrame from features DataFrame."""
    i = 0
    target_df = pd.DataFrame()
    while len(target_df.columns) == 0:
        target_pick = targets[i].lower()
        col_picks = [
            col for col in features_df.columns if target_pick in col.lower()
        ]
        for col in col_picks[:1]:
            target_df[col] = features_df[col]
        i += 1
    if verbose:
        print("Channel used: ", target_df.columns[0])
    return target_df

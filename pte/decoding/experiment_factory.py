"""Module for running decoding experiments."""
import os
import sys
from pathlib import Path
from joblib import Parallel, delayed
from typing import Iterable, Optional, Union

import numpy as np
import pandas as pd

import pte
from ..settings import PATH_PYNEUROMODULATION

sys.path.insert(0, PATH_PYNEUROMODULATION)
from py_neuromodulation.nm_analysis import Feature_Reader


def _run_single_experiment(
    feature_root: Union[Path, str],
    feature_file: Union[Path, str],
    classifier: str,
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
    plot_target_channels=None,
    artifact_channels=None,
    bad_events_path=None,
    pred_mode="classify",
    pred_begin=-3.0,
    pred_end=2.0,
    use_times=1,
    dist_onset=2.0,
    dist_end=2.0,
    excep_dist_end=0.5,
    exceptions=None,
    feature_importance=False,
    verbose=True,
) -> Optional[pte.decoding.Experiment]:
    """Run experiment with single file."""
    if verbose:
        print("Using file: ", feature_file)

    # Read features using py_neuromodulation
    nm_reader = Feature_Reader(
        feature_dir=feature_root, feature_file=feature_file
    )
    features = nm_reader.feature_arr
    settings = nm_reader.settings

    # Pick label for classification
    try:
        label = _get_label(label_channels, features, nm_reader)
    except ValueError as error:
        print(error, "Discarding file: {feature_file}")
        return None

    # Handle bad events file
    bad_events = pte.filetools.get_bad_events(bad_events_path, feature_file)

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

    dist_end = _handle_exception_files(
        fullpath=out_path,
        exception_files=exceptions,
        dist_end=dist_end,
        excep_dist_end=excep_dist_end,
    )

    side = "right" if "R_" in out_path else "left"

    decoder = pte.decoding.get_decoder(
        classifier=classifier,
        scoring=scoring,
        balancing=balancing,
        optimize=optimize,
    )

    kwargs = dict(
        scoring=scoring,
        feature_importance=feature_importance,
        target_begin=target_begin,
        target_end=target_end,
        dist_onset=dist_onset,
        dist_end=dist_end,
        use_channels=use_channels,
        pred_mode=pred_mode,
        pred_begin=pred_begin,
        pred_end=pred_end,
        cv_outer=cross_validation,
        verbose=verbose,
    )
    # Initialize Experiment instance
    experiment = pte.decoding.Experiment(
        features=features_df,
        target_df=target_df,
        label=label,
        ch_names=settings["ch_names"],
        decoder=decoder,
        side=side,
        artifacts=artifacts,
        bad_epochs=bad_events,
        sfreq=settings["sampling_rate_features"],
        **kwargs,
    )
    experiment.run()
    experiment.save(path=out_path)
    return experiment


def run_experiment(
    feature_root: Union[Path, str],
    feature_files: Union[Path, str, list[Union[Path, str]]],
    n_jobs: int = 1,
    **kwargs,
) -> list[pte.decoding.Experiment]:
    """Run prediction experiment with given number of files."""
    if not feature_files:
        raise ValueError("No feature files specified.")
    if not isinstance(feature_files, list):
        feature_files = [feature_files]
    if len(feature_files) == 1 or n_jobs in (0, 1):
        return [
            _run_single_experiment(
                feature_root=feature_root, feature_file=feature_file, **kwargs,
            )
            for feature_file in feature_files
        ]
    return [
        Parallel(n_jobs=n_jobs)(
            delayed(_run_single_experiment)(
                feature_root=feature_root, feature_file=feature_file, **kwargs
            )
            for feature_file in feature_files
        )
    ]


def _get_label(
    label_channels: list[str],
    features: pd.DataFrame,
    nm_reader: Feature_Reader,
) -> pd.Series:
    """Read label DataFrame from given file."""
    for label_channel in label_channels:
        if label_channel in features.columns:
            label_data = nm_reader.read_target_ch(
                feature_arr=features, label_name=label_channel, binarize=False,
            )
            return pd.Series(label_data, name=label_channel)
    raise ValueError(f"No valid label found. Labels given: {label_channels}.")


def _handle_exception_files(
    fullpath: str,
    exception_files: Optional[Iterable],
    dist_end: Union[int, float],
    excep_dist_end: Union[int, float],
):
    """Check if current file is listed in exception files."""
    if all(
        (exception_files, any(exc in fullpath for exc in exception_files),)
    ):
        print("Exception file recognized: ", os.path.basename(fullpath))
        return excep_dist_end
    return dist_end


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
    """Extract features to use from given DataFrame"""
    column_picks = [
        col
        for col in features.columns
        if any(pick in col for pick in use_features)
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

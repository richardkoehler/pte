"""Module for running decoding experiments."""
import os
import sys
from pathlib import Path
from joblib import Parallel, delayed
from typing import Iterable, Optional, Sequence, Union

import numpy as np
import pandas as pd

import pte
from .experiment import Experiment
from .decode import get_decoder

from py_neuromodulation import nm_analysis


def run_experiment(
    feature_root: Union[Path, str],
    feature_files: Union[Path, str, list[Union[Path, str]]],
    n_jobs: int = 1,
    **kwargs,
) -> list[Optional[Experiment]]:
    """Run prediction experiment with given number of files."""
    if not feature_files:
        raise ValueError("No feature files specified.")
    if not isinstance(feature_files, list):
        feature_files = [feature_files]
    if len(feature_files) == 1 or n_jobs in (0, 1):
        return [
            _run_single_experiment(
                feature_root=feature_root,
                feature_file=feature_file,
                **kwargs,
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
    ]  # type: ignore


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
    scoring: str = "balanced_accuracy",
    plot_target_channels: Optional[list[str]] = None,
    artifact_channels=None,
    bad_events_path=None,
    pred_mode: str = "classify",
    pred_begin: Union[int, float] = -3.0,
    pred_end: Union[int, float] = 2.0,
    use_times: int = 1,
    dist_onset: Union[int, float] = 2.0,
    dist_end: Union[int, float] = 2.0,
    excep_dist_end: Union[int, float] = 0.5,
    exceptions=None,
    feature_importance=False,
    verbose: bool = True,
) -> Optional[Experiment]:
    """Run experiment with single file."""
    if verbose:
        print("Using file: ", feature_file)

    # Read features using py_neuromodulation
    nm_reader = nm_analysis.Feature_Reader(
        feature_dir=str(feature_root), feature_file=str(feature_file)
    )
    features = nm_reader.feature_arr
    settings = nm_reader.settings
    sidecar = nm_reader.sidecar

    # Pick label for classification
    try:
        label = _get_column_picks(
            column_picks=label_channels,
            features=features,
        )
    except ValueError as error:
        print(error, "Discarding file: {feature_file}")
        return None

    # Handle bad events file
    bad_events = pte.filetools.get_bad_events(bad_events_path, feature_file)

    # Pick target for plotting predictions
    target_series = _get_column_picks(
        column_picks=plot_target_channels,
        features=features,
    )
    print("Target channels found:", target_series.name)

    features_df = _get_feature_df(features, use_features, use_times)

    # Pick artifact channel
    if artifact_channels:
        artifacts = _get_column_picks(
            column_picks=artifact_channels,
            features=features,
        ).to_numpy()
    else:
        artifacts = None
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

    side = "right" if "R_" in str(out_path) else "left"

    decoder = get_decoder(
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
    experiment = Experiment(
        features=features_df,
        target_df=target_series,
        label=label,
        ch_names=sidecar["ch_names"],
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


def _get_label(
    label_channels: list[str],
    features: pd.DataFrame,
    nm_reader: nm_analysis.Feature_Reader,
) -> pd.Series:
    """Read label DataFrame from given file."""
    for label_channel in label_channels:
        if label_channel in features.columns:
            label_data = nm_reader.read_target_ch(
                feature_arr=features,
                label_name=label_channel,
                binarize=False,
            )
            return pd.Series(label_data, name=label_channel)
    raise ValueError(f"No valid label found. Labels given: {label_channels}.")


def _handle_exception_files(
    fullpath: Union[Path, str],
    exception_files: Optional[Iterable],
    dist_end: Union[int, float],
    excep_dist_end: Union[int, float],
):
    """Check if current file is listed in exception files."""
    if exception_files:
        if any(exc in str(fullpath) for exc in exception_files):
            print("Exception file recognized: ", os.path.basename(fullpath))
            return excep_dist_end
    return dist_end


def _generate_outpath(
    root: Union[Path, str],
    feature_file: Union[Path, str],
    classifier: str,
    target_begin: Union[str, int, float],
    target_end: Union[str, int, float],
    use_channels: str,
    optimize: bool,
    use_times: int,
) -> Path:
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
    return Path(root, out_name, feature_file, feature_file)


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


def _get_column_picks(
    column_picks: Optional[Sequence[str]],
    features: pd.DataFrame,
) -> pd.Series:
    """Return first found column pick from features DataFrame."""
    if column_picks:
        for pick in column_picks:
            for col in features.columns:
                if pick.lower() in col.lower():
                    return pd.Series(data=features[col], name=col)
    raise ValueError(
        f"No valid column found. `column_picks` given: {column_picks}."
    )

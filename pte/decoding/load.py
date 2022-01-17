"""Module for loading results from decoding experiments."""
import json
from pathlib import Path
from typing import Optional, Union

import mne_bids
import numpy as np
import pandas as pd

import pte


def load_results_singlechannel(
    files_or_dir: Union[str, list, Path],
    keywords: Optional[Union[str, list]] = None,
    scoring_key: str = "balanced_accuracy",
    average_runs: bool = False,
) -> pd.DataFrame:
    """Load results from *results.csv"""
    # Create Dataframes from Files
    if not isinstance(files_or_dir, list):
        file_finder = pte.get_filefinder(datatype="any")
        file_finder.find_files(
            directory=files_or_dir,
            keywords=keywords,
            extensions=["results.csv"],
            verbose=True,
        )
        files_or_dir = file_finder.files
    results = []
    for file in files_or_dir:
        subject = mne_bids.get_entities_from_fname(file, on_error="ignore")[
            "subject"
        ]
        data: pd.DataFrame = pd.read_csv(
            file,
            index_col="channel_name",
            header=0,
            usecols=["channel_name", scoring_key],
        )
        for ch_name in data.index.unique():
            score = data.loc[ch_name].mean(numeric_only=True).values[0]  # type: ignore
            results.append([subject, ch_name, score])
    columns = [
        "Subject",
        "Channel Name",
        scoring_key,
    ]
    data_out = pd.DataFrame(results, columns=columns)
    if average_runs:
        data_out = data_out.set_index(
            keys=["Subject", "Channel Name"]
        ).sort_index()
        results_average = []
        for ind in data_out.index.unique():
            result_av = data_out.loc[ind].mean(numeric_only=True).values[0]
            results_average.append([*ind, result_av])
        data_out = pd.DataFrame(results_average, columns=columns)
    return data_out


def load_results(
    files_or_dir: Union[str, list, Path],
    keywords: Optional[Union[str, list]] = None,
    scoring_key: str = "balanced_accuracy",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load prediciton results from *results.csv"""
    # Create Dataframes from Files
    if not isinstance(files_or_dir, list):
        file_finder = pte.get_filefinder(datatype="any")
        file_finder.find_files(
            directory=files_or_dir,
            keywords=keywords,
            extensions=["results.csv"],
            verbose=True,
        )
        files_or_dir = file_finder.files
    results = []
    for file in files_or_dir:
        data_raw = pd.read_csv(file, index_col=[0], header=[0])
        data: pd.DataFrame = pd.melt(
            data_raw, id_vars=["channel_name"], value_vars=[scoring_key]
        )
        accuracies = []
        for ch_name in data["channel_name"].unique():
            accuracies.append(
                [
                    "LFP" if "LFP" in ch_name else "ECOG",
                    data[data.channel_name == ch_name]
                    .mean(numeric_only=True)
                    .value,  # type: ignore
                ]
            )
        df_acc = pd.DataFrame(accuracies, columns=["Channels", scoring_key])
        df_lfp = df_acc[df_acc["Channels"] == "LFP"]
        df_ecog = df_acc[df_acc["Channels"] == "ECOG"]
        subject = mne_bids.get_entities_from_fname(file, on_error="ignore")[
            "subject"
        ]
        values = [
            file,
            subject,
            "OFF" if "MedOff" in file else "ON",
            "OFF" if "StimOff" in file else "ON",
        ]
        results.extend(
            [
                values + ["LFP", df_lfp[scoring_key].max()],
                values + ["ECOG", df_ecog[scoring_key].max()],
            ]
        )
    columns = [
        "filename",
        "subject",
        "medication",
        "stimulation",
        "channels",
        scoring_key,
    ]
    df_raw = pd.DataFrame(results, columns=columns)

    # Average raw results
    results_average = []
    for ch_name in df_raw["channels"].unique():
        df_ch = df_raw.loc[df_raw["channels"] == ch_name]
        for subject in df_ch["subject"].unique():
            df_subj = df_ch.loc[df_ch["subject"] == subject]
            series_single = pd.Series(
                df_subj.iloc[0].values, index=df_subj.columns
            ).drop("filename")
            series_single[scoring_key] = df_subj[scoring_key].mean()
            results_average.append(series_single)
    df_average = pd.DataFrame(results_average)

    # Rename columns
    df_average = df_average.rename(
        columns={
            col: " ".join([substr.capitalize() for substr in col.split("_")])
            for col in df_average.columns
        }
    )
    return df_average, df_raw


def _handle_files_or_dir(
    files_or_dir: Union[str, list, Path],
    keywords: Optional[Union[str, list]] = None,
) -> list:
    """Handle different cases of files_or_dir."""
    if isinstance(files_or_dir, list):
        return files_or_dir
    file_finder = pte.get_filefinder(datatype="any")
    file_finder.find_files(
        directory=files_or_dir,
        keywords=keywords,
        extensions=["predictions_timelocked.json"],
        verbose=True,
    )
    return file_finder.files


def _load_labels_single(
    fpath: Union[str, Path],
    baseline: Optional[
        tuple[Optional[Union[int, float]], Optional[Union[int, float]]]
    ],
    baseline_mode: Optional[str],
    base_start: Optional[int],
    base_end: Optional[int],
) -> pd.DataFrame:
    """Load time-locked predictions from single file."""
    entities = mne_bids.get_entities_from_fname(fpath, on_error="ignore")
    with open(fpath, "r", encoding="utf-8") as file:
        data = json.load(file)

    label_name = data["TargetName"]
    label_arr = np.stack(data["Target"], axis=0)

    if baseline is not None:
        label_arr = _baseline_correct(
            label_arr, baseline_mode, base_start, base_end
        )

    labels = pd.DataFrame(
        data=[
            [
                entities["subject"],
                entities["session"],
                entities["task"],
                entities["run"],
                entities["acquisition"],
                label_name,
                label_arr,
            ]
        ],
        columns=[
            "Subject",
            "Session",
            "Task",
            "Run",
            "Acquisition",
            "Channel Name",
            "Data",
        ],
    )
    return labels


def load_predictions(
    files_or_dir: Union[str, list, Path],
    mode: str = "predictions",
    sfreq: Optional[Union[int, float]] = None,
    baseline: Optional[
        tuple[Optional[Union[int, float]], Optional[Union[int, float]]]
    ] = None,
    baseline_mode: str = "z-score",
    average_predictions: bool = False,
    concatenate_runs: bool = True,
    average_runs: bool = False,
    keywords: Optional[Union[str, list]] = None,
) -> pd.DataFrame:
    """Load time-locked predictions."""
    files_or_dir = _handle_files_or_dir(
        files_or_dir=files_or_dir, keywords=keywords
    )

    base_start, base_end = _handle_baseline(baseline=baseline, sfreq=sfreq)

    df_list = []
    for fpath in files_or_dir:
        if mode == "predictions":
            df_single: pd.DataFrame = _load_predictions_single(
                fpath=fpath,
                baseline=baseline,
                baseline_mode=baseline_mode,
                base_start=base_start,
                base_end=base_end,
            )
        elif mode == "targets":
            df_single: pd.DataFrame = _load_labels_single(
                fpath=fpath,
                baseline=baseline,
                baseline_mode=baseline_mode,
                base_start=base_start,
                base_end=base_end,
            )
        else:
            raise ValueError(
                "`mode` must be one of either `targets` or "
                f"`predictions. Got: {mode}."
            )
        if average_predictions:
            df_single["Data"] = (
                df_single["Data"]
                .apply(np.mean, axis=0)
                .apply(np.expand_dims, axis=0)
            )
        df_list.append(df_single)
    df_all: pd.DataFrame = pd.concat(objs=df_list)
    if concatenate_runs:
        df_all = _concatenate_runs(data=df_all)
    if average_runs:
        df_all["Data"] = (
            df_all["Data"].apply(np.mean, axis=0).apply(np.expand_dims, axis=0)
        )
    return df_all


def _concatenate_runs(data: pd.DataFrame):
    """Concatenate predictions from different runs in a single patient."""
    data_list = []
    for subject in data["Subject"].unique():
        dat_sub = data[data["Subject"] == subject]
        for ch_name in dat_sub["Channel Name"].unique():
            dat_concat = np.vstack(
                dat_sub["Data"][dat_sub["Channel Name"] == ch_name].to_numpy()
            )
            data_list.append([subject, ch_name, dat_concat])
    return pd.DataFrame(data_list, columns=["Subject", "Channel Name", "Data"])


def _load_predictions_single(
    fpath: Union[str, Path],
    baseline: Optional[
        tuple[Optional[Union[int, float]], Optional[Union[int, float]]]
    ],
    baseline_mode: str,
    base_start: Optional[int],
    base_end: Optional[int],
) -> pd.DataFrame:
    """Load time-locked predictions from single file."""
    entities = mne_bids.get_entities_from_fname(fpath, on_error="ignore")
    with open(fpath, "r", encoding="utf-8") as file:
        preds = json.load(file)
    data_list = []
    for ch_name in preds.keys():
        if any(keyw in ch_name for keyw in ["ECOG", "LFP"]):
            pred_arr = np.stack(preds[ch_name], axis=0)
            if baseline:
                pred_arr = _baseline_correct(
                    pred_arr, baseline_mode, base_start, base_end
                )
            data_list.append(
                [
                    entities["subject"],
                    entities["session"],
                    entities["task"],
                    entities["run"],
                    entities["acquisition"],
                    ch_name,
                    pred_arr,
                ]
            )
    data = pd.DataFrame(
        data=data_list,
        columns=[
            "Subject",
            "Session",
            "Task",
            "Run",
            "Acquisition",
            "Channel Name",
            "Data",
        ],
    )
    return data


def _handle_baseline(
    baseline: Optional[
        tuple[Optional[Union[int, float]], Optional[Union[int, float]]]
    ],
    sfreq: Optional[Union[int, float]],
) -> tuple[Optional[int], Optional[int]]:
    """Return baseline start and end values for given baseline and sampling frequency."""
    if not baseline:
        return None, None
    if any(baseline) and not sfreq:
        raise ValueError(
            "If `baseline` is any value other than `None`, or `(None, None)`, `sfreq` must be provided."
        )
    if not sfreq:
        sfreq = 0.0
    if baseline[0] is None:
        base_start = 0
    else:
        base_start = int(baseline[0] * sfreq)
    if baseline[1] is None:
        base_end = None
    else:
        base_end = int(baseline[1] * sfreq)
    return base_start, base_end


def _baseline_correct(
    data: np.ndarray,
    baseline_mode: str,
    base_start: Optional[int],
    base_end: Optional[int],
) -> np.ndarray:
    """Baseline correct data."""
    data = data.T
    if baseline_mode == "zscore":
        data = (data - np.mean(data[base_start:base_end], axis=0)) / (
            np.std(data[base_start:base_end], axis=0)
        )
        return data.T
    if baseline_mode == "std":
        data = data / np.std(data[base_start:base_end], axis=0)
        return data.T
    raise ValueError(
        "`baseline_mode` must be one of either `std` or `zscore`."
        f" Got: {baseline_mode}."
    )

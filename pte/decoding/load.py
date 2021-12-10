"""Module for loading results from decoding experiments."""
import json
from pathlib import Path
from typing import Iterable, Optional, Union

import numpy as np
import pte


def load_predictions_timelocked(
    fpath_or_dir: Union[str, list, Path],
    sfreq: Optional[Union[int, float]] = None,
    baseline: Union[bool, tuple] = None,
    channels: Iterable = ("ECOG", "LFP"),
    keywords: Optional[Union[str, list]] = None,
    average: Union[bool, int] = False,
):
    """Load data from time-locked predictions"""
    if baseline is not None:
        if any(baseline) and not sfreq:
            raise ValueError(
                "If `baseline` is any value other than `None` or `(None, None)`, `sfreq` must be provided."
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

    if not isinstance(fpath_or_dir, list):
        file_finder = pte.get_filefinder(datatype="any")
        file_finder.find_files(
            directory=fpath_or_dir,
            keywords=keywords,
            extensions=["predictions_timelocked.json"],
            verbose=True,
        )
        fpath_or_dir = file_finder.files
    data = []
    for fpath in fpath_or_dir:
        with open(fpath, "r", encoding="utf-8") as file:
            preds = json.load(file)
        data_single = []
        for ch_name in channels:
            pred = np.mean(np.stack(preds[ch_name], axis=0), axis=0)
            if baseline:
                if baseline[1] is None:
                    base_end = pred.shape[0]
                pred = (pred - np.mean(pred[base_start:base_end])) / (
                    np.std(pred[base_start:base_end])
                )
            data_single.append(pred)
        data.append(data_single)
    data = np.array(data)
    if average:
        if average is True:
            average = 0
        data = data.mean(axis=average)
    return data

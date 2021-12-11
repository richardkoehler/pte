"""Module for loading results from decoding experiments."""
import json
from pathlib import Path
from typing import Iterable, Optional, Union
import mne_bids

import numpy as np
import pte


def load_predictions_timelocked(
    fpath_or_dir: Union[str, list, Path],
    sfreq: Optional[Union[int, float]] = None,
    baseline: Union[bool, tuple] = None,
    channels: Iterable = ("ECOG", "LFP"),
    keywords: Optional[Union[str, list]] = None,
    key_average: Optional[str] = None,
):
    """Load data from time-locked predictions."""
    if not isinstance(fpath_or_dir, list):
        file_finder = pte.get_filefinder(datatype="any")
        file_finder.find_files(
            directory=fpath_or_dir,
            keywords=keywords,
            extensions=["predictions_timelocked.json"],
            verbose=True,
        )
        fpath_or_dir = file_finder.files
    if baseline:
        if any(baseline) and not sfreq:
            raise ValueError(
                "If `baseline` is any value other than `None`, `False` or `(None, None)`, `sfreq` must be provided."
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

    data = {ch_name: {} for ch_name in channels}
    for key, fpath in enumerate(fpath_or_dir):
        if key_average:
            key = mne_bids.get_entities_from_fname(fpath, on_error="ignore")[
                key_average
            ]
        with open(fpath, "r", encoding="utf-8") as file:
            preds = json.load(file)
        for ch_name in channels:
            if key not in data[ch_name]:
                data[ch_name][key] = []
            pred = np.mean(np.stack(preds[ch_name], axis=0), axis=0)
            if baseline:
                if baseline[1] is None:
                    base_end = pred.shape[0]
                pred = (pred - np.mean(pred[base_start:base_end])) / (
                    np.std(pred[base_start:base_end])
                )
            data[ch_name][key].append(pred)
    data_outer = []
    for k, value in data.items():
        data_inner = []
        for i, j in value.items():
            data_inner.append(np.array(j).mean(axis=0))
        data_outer.append(data_inner)
    data_outer = np.array(data_outer)
    return data_outer

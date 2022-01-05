"""Module for power calculation."""


import inspect
from pathlib import Path
from typing import Optional, Union

import mne
import mne_bids
import numpy as np

import pte


def morlet_from_epochs(
    epochs: mne.Epochs,
    n_cycles: int = 7,
    freqs: Optional[np.ndarray] = None,
    average: bool = True,
    n_jobs: int = -1,
    picks="all",
    **kwargs,
) -> Union[mne.time_frequency.AverageTFR, mne.time_frequency.EpochsTFR]:
    """Calculate power with MNE's Morlet transform and sensible defaults."""
    if freqs is None:
        freqs = np.arange(1, epochs.info["sfreq"])
    power = mne.time_frequency.tfr_morlet(
        inst=epochs,
        freqs=freqs,
        n_cycles=n_cycles,
        n_jobs=n_jobs,
        picks=picks,
        average=average,
        return_itc=False,
        **kwargs,
    )
    return power


def epochs_from_raw(
    raw: mne.io.BaseRaw,
    event_picks: Optional[list[str]],
    tmin: Union[int, float] = -6,
    tmax: Union[int, float] = 6,
    baseline: Optional[tuple] = None,
    **kwargs,
) -> mne.Epochs:
    """Return epochs from given events."""
    events = None
    for event_pick in event_picks:
        try:
            events, event_id = mne.events_from_annotations(
                raw=raw, event_id={event_pick: 1}, verbose=True,
            )
            break
        except ValueError:
            pass
    if events is None:
        _, event_id = mne.events_from_annotations(raw=raw, verbose=False,)
        raise ValueError(
            f"None of the given `event_picks´ found: {event_picks}."
            f"Possible events: {*event_id.keys(),}"
        )
    print(event_id)
    epochs = mne.Epochs(
        raw=raw,
        events=events,
        tmin=tmin,
        tmax=tmax,
        baseline=baseline,
        **kwargs,
    )
    return epochs


def power_from_bids(
    file: mne_bids.BIDSPath,
    event_picks: Optional[list[str]],
    out_dir: Optional[Union[Path, str]] = None,
    bad_events_dir: Optional[Union[Path, str]] = None,
    **kwargs,
) -> mne.time_frequency.AverageTFR:
    """Calculate power from single file."""
    print(f"File: {file.basename}")
    raw = mne_bids.read_raw_bids(file, verbose=False)

    decim_power = "auto"
    if "decim" in kwargs:
        decim_power = kwargs.pop("decim")

    args_preprocess = {"raw": raw}
    for key in inspect.getfullargspec(pte.processing.preprocess)[0]:
        if key in kwargs:
            args_preprocess[key] = kwargs[key]
    raw = pte.processing.preprocess(**args_preprocess)

    kwargs_epochs = {"event_picks": event_picks, "baseline": None}
    for key in inspect.getfullargspec(mne.Epochs)[0]:
        if key in kwargs:
            kwargs_epochs[key] = kwargs[key]
    epochs = epochs_from_raw(raw=raw, **kwargs_epochs)

    if bad_events_dir:
        bad_events = pte.filetools.get_bad_events(
            bad_events_path=bad_events_dir, fname=file
        )
        if bad_events is not None:
            epochs = epochs.drop(indices=bad_events)

    if decim_power == "auto":
        decim = int(epochs.info["sfreq"] / 100)
    else:
        decim = decim_power
    kwargs_power = {"decim": decim}
    for key in inspect.getfullargspec(mne.time_frequency.tfr_morlet)[0]:
        if key in kwargs:
            kwargs_power[key] = kwargs[key]
        if "freqs" not in kwargs_power:
            kwargs_power["freqs"] = np.arange(1, 200, 1)

    power = morlet_from_epochs(epochs=epochs, **kwargs_power,)
    if out_dir:
        fname = Path(out_dir) / (str(Path(file).stem) + "_tfr.h5")
        power.save(fname=fname, verbose=True)
    return power


def power_from_files(
    filenames: list[mne_bids.BIDSPath],
    event_picks: Optional[list[str]],
    out_dir: Optional[Union[Path, str]] = None,
    bad_events_dir: Optional[Union[Path, str]] = None,
    decim: Union[int, str] = "auto",
    **kwargs,
) -> None:
    """Perform Morlet transform on batch of given BIDS files."""
    if not filenames:
        raise ValueError("No filenames given.")
    return [
        power_from_bids(
            file=file,
            event_picks=event_picks,
            out_dir=out_dir,
            bad_events_dir=bad_events_dir,
            decim=decim,
            **kwargs,
        )
        for file in filenames
    ]

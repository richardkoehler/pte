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
        verbose=True,
        **kwargs,
    )
    return power


def epochs_from_raw(
    raw: mne.io.BaseRaw,
    tmin: Union[int, float] = -6,
    tmax: Union[int, float] = 6,
    baseline: Optional[tuple] = None,
    events_trial_onset: Optional[
        Union[str, list[str], list[tuple[str, str]]]
    ] = None,
    events_trial_end: Optional[
        Union[str, list[str], list[tuple[str, str]]]
    ] = None,
    min_distance_trials: Union[int, float] = 0,
    **kwargs,
) -> mne.Epochs:
    """Return epochs from given events."""
    events, _ = get_events(raw=raw, event_picks=events_trial_onset)
    epochs = mne.Epochs(
        raw=raw,
        events=events,
        tmin=tmin,
        tmax=tmax,
        baseline=baseline,
        verbose=True,
        **kwargs,
    )
    if min_distance_trials:
        events_end = None
        if events_trial_end:
            events_end, _ = get_events(raw, event_picks=events_trial_end)
        epochs = discard_epochs(
            epochs=epochs,
            events_begin=events,
            events_end=events_end,
            min_distance_events=min_distance_trials,
        )
    return epochs


def get_events(
    raw: mne.io.Raw, event_picks: Union[str, list[str], list[tuple[str, str]]]
) -> tuple[np.ndarray, dict]:
    """Get events from given Raw instance and event id."""
    if isinstance(event_picks, str):
        event_picks = [event_picks]
    events = None
    for event_pick in event_picks:
        if isinstance(event_pick, str):
            event_id = {event_pick: 1}
        else:
            event_id = {event_pick[0]: 1, event_pick[1]: -1}
        try:
            events, event_id = mne.events_from_annotations(
                raw=raw, event_id=event_id, verbose=True,
            )
            break
        except ValueError:
            pass
    if events is None:
        _, event_id_found = mne.events_from_annotations(
            raw=raw, verbose=False,
        )
        raise ValueError(
            f"None of the given `event_picks´ found: {event_picks}."
            f"Possible events: {*event_id_found.keys(),}"
        )
    return events, event_id


def discard_epochs(
    epochs: mne.Epochs,
    events_begin: np.ndarray,
    min_distance_events: Union[int, float],
    events_end: Optional[np.ndarray] = None,
) -> mne.Epochs:
    """Discard epochs based on minimal distance between event onset and end."""
    if events_end is not None:
        events = np.sort(np.hstack((events_begin[:, 0], events_end[:, 0])))
        event_diffs = np.diff(events)[1::2]
    else:
        events = events_begin[:, 0]
        event_diffs = np.diff(events)
    drop_indices = np.where(
        event_diffs <= min_distance_events * epochs.info["sfreq"]
    )[0]
    epochs = epochs.drop(indices=drop_indices)
    return epochs


def power_from_bids(
    file: mne_bids.BIDSPath,
    events_trial_onset: Optional[list[str]] = None,
    events_trial_end: Optional[list[str]] = None,
    min_distance_trials: Union[int, float] = 0,
    bad_events_dir: Optional[Union[Path, str]] = None,
    out_dir: Optional[Union[Path, str]] = None,
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

    args_epochs = {"raw": raw}
    for key in inspect.getfullargspec(mne.Epochs)[0]:
        if key in kwargs:
            args_epochs[key] = kwargs[key]
    epochs = epochs_from_raw(
        **args_epochs,
        events_trial_onset=events_trial_onset,
        events_trial_end=events_trial_end,
        min_distance_trials=min_distance_trials,
    )

    if bad_events_dir:
        bad_events = pte.filetools.get_bad_events(
            bad_events_path=bad_events_dir, fname=file,
        )
        if bad_events is not None:
            bad_indices = np.array(
                [
                    idx
                    for idx, event in enumerate(epochs.selection)
                    if event in bad_events
                ]
            )
            epochs = epochs.drop(indices=bad_indices)

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
    events_trial_onset: Optional[list[str]],
    out_dir: Optional[Union[Path, str]] = None,
    bad_events_dir: Optional[Union[Path, str]] = None,
    min_distance_trials: Union[int, float] = 0,
    decim: Union[int, str] = "auto",
    **kwargs,
) -> None:
    """Perform Morlet transform on batch of given BIDS files."""
    if not filenames:
        raise ValueError("No filenames given.")
    return [
        power_from_bids(
            file=file,
            events_trial_onset=events_trial_onset,
            out_dir=out_dir,
            bad_events_dir=bad_events_dir,
            min_distance_trials=min_distance_trials,
            decim=decim,
            **kwargs,
        )
        for file in filenames
    ]

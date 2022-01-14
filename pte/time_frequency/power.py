"""Module for power calculation."""


import inspect
from pathlib import Path
from typing import Optional, Union

import mne
import mne_bids
import numpy as np

import pte


def average_power(
    powers: Union[
        list[mne.time_frequency.AverageTFR],
        list[mne.time_frequency.AverageTFR],
    ],
    picks: Union[str, list[str], slice],
    baseline: Optional[
        tuple[Optional[Union[int, float]], Optional[Union[int, float]]]
    ] = (None, None),
    baseline_mode: Optional[str] = "zscore",
    clip: Optional[Union[int, float]] = None,
) -> mne.time_frequency.AverageTFR:
    """Return power averaged over given channel types or picks."""
    if not isinstance(powers, list):
        powers = [powers]
    power_all = None
    power_all_files = []
    for power in powers:
        power = power.copy().pick(picks=picks)
        if baseline:
            power = power.apply_baseline(
                baseline=baseline, mode=baseline_mode, verbose=False
            )
        df_power = power.to_data_frame(picks=picks)
        freqs = power.freqs
        power_all_freqs = []
        for freq in freqs:
            power_single_freq = (
                df_power[df_power["freq"] == freq]
                .sort_values(by="time", axis=0)
                .drop(columns=["freq", "time"])
                .to_numpy()
            )
            # reject artifacts by clipping
            if clip is not None:
                power_single_freq = power_single_freq.clip(min=-clip, max=clip)
            power_all_freqs.append(power_single_freq)
        # Average across all channels
        power_all_files.append(np.stack(power_all_freqs, axis=0).mean(axis=-1))
    power_array_all = np.expand_dims(
        np.stack(power_all_files, axis=0).mean(axis=0), axis=0
    )
    if clip:
        power_array_all = power_array_all.clip(min=-clip, max=clip)
    power = powers[0]
    info = mne.create_info(
        ch_names=1, sfreq=power.info["sfreq"], ch_types="misc", verbose=False
    )
    power_all = mne.time_frequency.AverageTFR(
        info=info,
        data=power_array_all,
        times=power.times,
        freqs=power.freqs,
        nave=len(powers),
        comment=power.comment,
        method=power.method,
    )
    return power_all


def load_power(
    files: list[Union[Path, str]]
) -> Union[
    list[mne.time_frequency.AverageTFR], list[mne.time_frequency.EpochsTFR]
]:
    """Load power from *-tfr.h5 files."""
    powers = []
    for file in files:
        powers.append(mne.time_frequency.read_tfrs(file, condition=0))
    return powers


def morlet_from_epochs(
    epochs: mne.Epochs,
    n_cycles: int = 7,
    freqs: Union[np.ndarray, str] = "auto",
    average: bool = True,
    n_jobs: int = -1,
    picks="all",
    decim_power: Union[str, int, float] = "auto",
    **kwargs,
) -> Union[mne.time_frequency.AverageTFR, mne.time_frequency.EpochsTFR]:
    """Calculate power with MNE's Morlet transform and sensible defaults."""
    if freqs is None:
        upper_freq = min(epochs.info["sfreq"] / 2, 200.0)
        freqs = np.arange(1.0, upper_freq)

    if decim_power == "auto":
        decim = int(epochs.info["sfreq"] / 100)
    else:
        decim = decim_power

    power = mne.time_frequency.tfr_morlet(
        inst=epochs,
        freqs=freqs,
        n_cycles=n_cycles,
        n_jobs=n_jobs,
        picks=picks,
        average=average,
        return_itc=False,
        verbose=True,
        decim=decim,
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
    raw: mne.io.BaseRaw,
    event_picks: Union[str, list[str], list[tuple[str, str]]],
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
            events, _ = mne.events_from_annotations(
                raw=raw,
                event_id=event_id,
                verbose=True,
            )
            return events, event_id
        except ValueError as error:
            print(error)
    _, event_id_found = mne.events_from_annotations(
        raw=raw,
        verbose=False,
    )
    raise ValueError(
        f"None of the given `event_picks´ found: {event_picks}."
        f"Possible events: {*event_id_found.keys(),}"
    )


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
    nm_channels_dir: Path,
    events_trial_onset: Optional[list[str]] = None,
    events_trial_end: Optional[list[str]] = None,
    min_distance_trials: Union[int, float] = 0,
    bad_events_dir: Optional[Union[Path, str]] = None,
    out_dir: Optional[Union[Path, str]] = None,
    kwargs_preprocess: Optional[dict] = None,
    kwargs_epochs: Optional[dict] = None,
    kwargs_power: Optional[dict] = None,
) -> Union[mne.time_frequency.AverageTFR, mne.time_frequency.EpochsTFR]:
    """Calculate power from single file."""
    print(f"File: {file.basename}")
    raw = mne_bids.read_raw_bids(file, verbose=False)

    if kwargs_preprocess:
        raw = pte.processing.preprocess(
            raw=raw,
            nm_channels_dir=nm_channels_dir,
            **kwargs_preprocess,
        )
    else:
        raw = pte.processing.preprocess(
            raw=raw,
            nm_channels_dir=nm_channels_dir,
        )

    if kwargs_epochs:
        epochs = epochs_from_raw(
            raw=raw,
            events_trial_onset=events_trial_onset,
            events_trial_end=events_trial_end,
            min_distance_trials=min_distance_trials,
            **kwargs_epochs,
        )
    else:
        epochs = epochs_from_raw(
            raw=raw,
            events_trial_onset=events_trial_onset,
            events_trial_end=events_trial_end,
            min_distance_trials=min_distance_trials,
        )

    if bad_events_dir:
        bad_events = pte.filetools.get_bad_events(
            bad_events_path=bad_events_dir,
            fname=file.fpath,
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

    if not kwargs_power or "freqs" not in kwargs_power:
        freqs = np.arange(1, 200, 1)
    else:
        freqs = kwargs_power.pop("freqs")
    if kwargs_power:
        power = morlet_from_epochs(
            epochs=epochs,
            freqs=freqs,
            **kwargs_power,
        )
    else:
        power = morlet_from_epochs(
            epochs=epochs,
            freqs=freqs,
        )

    if out_dir:
        fname = Path(out_dir) / (str(file.fpath.stem) + "_tfr.h5")
        power.save(fname=fname, verbose=True)
    return power


def power_from_files(
    filenames: list[mne_bids.BIDSPath],
    nm_channels_dir,
    events_trial_onset: Optional[list[str]],
    out_dir: Optional[Union[Path, str]] = None,
    bad_events_dir: Optional[Union[Path, str]] = None,
    min_distance_trials: Union[int, float] = 0,
    kwargs_preprocess: Optional[dict] = None,
    kwargs_epochs: Optional[dict] = None,
    kwargs_power: Optional[dict] = None,
    **kwargs,
) -> list[Union[mne.time_frequency.AverageTFR, mne.time_frequency.EpochsTFR]]:
    """Perform Morlet transform on batch of given BIDS files."""
    if not filenames:
        raise ValueError("No filenames given.")
    return [
        power_from_bids(
            file=file,
            nm_channels_dir=nm_channels_dir,
            events_trial_onset=events_trial_onset,
            out_dir=out_dir,
            bad_events_dir=bad_events_dir,
            min_distance_trials=min_distance_trials,
            kwargs_preprocess=kwargs_preprocess,
            kwargs_epochs=kwargs_epochs,
            kwargs_power=kwargs_power,
            **kwargs,
        )
        for file in filenames
    ]

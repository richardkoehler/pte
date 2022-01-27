"""Functions for calculating and handling band-power."""
from pathlib import Path
from typing import Optional, Union

import matplotlib.figure
from matplotlib import pyplot as plt
import mne
import mne_bids
import numpy as np

import pte


def plot_power_diff(
    power_0: mne.time_frequency.AverageTFR,
    power_1: mne.time_frequency.AverageTFR,
    title: Optional[str] = None,
    fname: Optional[Union[str, Path]] = None,
    show: bool = True,
    kwargs_plot: Optional[dict] = None,
) -> matplotlib.figure.Figure:
    """Plot difference of two MNE AverageTFR objects."""
    power = power_0 - power_1
    fig, axes = plt.subplots(
        nrows=1,
        ncols=1,
        figsize=(4.75, 3.75),
        tight_layout=True,
    )
    if not kwargs_plot:
        kwargs_plot = {}
    kwargs_plot.update(
        picks="all",
        cmap="viridis",
        title=title,
        show=show,
        axes=axes,
        verbose=False,
    )
    fig = power.plot(
        **kwargs_plot,
    )[0]
    if show:
        plt.show(block=True)
    fig.savefig(
        fname=fname,
        dpi=300,
        bbox_inches="tight",
    )
    return fig


def plot_power(
    power: mne.time_frequency.AverageTFR,
    title: Optional[str] = None,
    fname: Optional[Union[str, Path]] = None,
    show: bool = True,
    kwargs_plot: Optional[dict] = None,
) -> matplotlib.figure.Figure:
    """Plot single MNE AverageTFR object."""
    fig, axes = plt.subplots(
        nrows=1,
        ncols=1,
        figsize=(4.75, 3.75),
        tight_layout=True,
    )
    if not kwargs_plot:
        kwargs_plot = {}
    kwargs_plot.update(
        picks="all",
        cmap="viridis",
        title=title,
        axes=axes,
        show=show,
        verbose=False,
    )
    fig = power.plot(
        **kwargs_plot,
    )[0]
    if show:
        plt.show(block=True)
    if fname:
        fig.savefig(
            fname=fname,
            dpi=300,
            bbox_inches="tight",
        )
    return fig


def apply_baseline(
    power: mne.time_frequency.AverageTFR,
    baseline: Optional[
        Union[
            tuple[Optional[Union[int, float]], Optional[Union[int, float]]],
            np.ndarray,
        ]
    ] = (None, None),
    baseline_mode: str = "zscore",
):
    """Apply baseline correction given interval or custom values."""
    if not isinstance(baseline, np.ndarray):
        return power.apply_baseline(
            baseline=baseline, mode=baseline_mode, verbose=False
        )
    return _apply_baseline_array(
        power=power, baseline=baseline, mode=baseline_mode
    )


# This code is adapted from MNE-Python version 0.24.1 (mne.baseline.rescale())
def _apply_baseline_array(
    power: mne.time_frequency.AverageTFR,
    baseline: np.ndarray,
    mode: str,
) -> mne.time_frequency.AverageTFR:
    """Apply baseline correction given array."""
    data = power.data.copy()
    test_2 = data.copy()
    if not baseline.ndim == data.ndim:
        raise ValueError(
            "If `baseline` is an array of values, it must have"
            " the same number of dimensions as `power`. Number of dimensions"
            f" `baseline`: {baseline.ndim}. Number of dimensions `power`:"
            f" {data.ndim}."
        )
    mean = np.mean(data, axis=-1, keepdims=True)

    if mode == "mean":

        def fun(d, m):
            d -= m

    elif mode == "ratio":

        def fun(d, m):
            d /= m

    elif mode == "logratio":

        def fun(d, m):
            d /= m
            np.log10(d, out=d)

    elif mode == "percent":

        def fun(d, m):
            d -= m
            d /= m

    elif mode == "zscore":

        def fun(d, m):
            d -= m
            d /= np.std(d, axis=-1, keepdims=True)

    elif mode == "zlogratio":

        def fun(d, m):
            d /= m
            np.log10(d, out=d)
            d /= np.std(d, axis=-1, keepdims=True)

    else:
        raise ValueError(f"Unknown baseline correction mode: {mode}.")

    fun(data, mean)

    power.data = power.data - data

    return power


# This code is adapted from MNE-Python version 0.24.1 (mne.baseline.rescale())
def _get_baseline_indices(
    baseline: tuple[Optional[Union[int, float]], Optional[Union[int, float]]],
    times: np.ndarray,
) -> tuple[int, int]:
    """Get baseline indices from times array."""
    bmin, bmax = baseline
    if bmin is None:
        imin = 0
    else:
        imin = np.where(times >= bmin)[0]
        if len(imin) == 0:
            raise ValueError(
                f"bmin is too large {bmin}, it exceeds the largest"
                " time value."
            )
        imin = int(imin[0])
    if bmax is None:
        imax = len(times)
    else:
        imax = np.where(times <= bmax)[0]
        if len(imax) == 0:
            raise ValueError(
                f"bmax is too small {bmax}, it is smaller than the"
                " smallest time value."
            )
        imax = int(imax[-1]) + 1
    if imin >= imax:
        raise ValueError(
            f"Bad rescaling slice ({imin}:{imax}) from time values"
            f" {bmin}, {bmax}."
        )
    return imin, imax


def get_baseline(
    powers: list[mne.time_frequency.AverageTFR],
    picks: Union[str, list[str], slice],
    baseline: tuple[
        Optional[Union[int, float]], Optional[Union[int, float]]
    ] = (None, None),
) -> list[np.ndarray]:
    """Get baseline array of given list of AverageTFRs."""
    if not isinstance(powers, list):
        powers = [powers]

    baselines = []
    for power in powers:
        baseline_array = power.copy().pick(picks=picks)
        imin, imax = _get_baseline_indices(
            baseline=baseline, times=power.times
        )
        baseline_array = power.data[..., imin:imax]
        baselines.append(baseline_array)

    return baselines


def average_power(
    powers: Union[
        list[mne.time_frequency.AverageTFR], mne.time_frequency.AverageTFR
    ],
    picks: Union[str, list[str], slice],
    baseline: Optional[
        Union[
            tuple[Optional[Union[int, float]], Optional[Union[int, float]]],
            list[np.ndarray],
        ]
    ] = (None, None),
    baseline_mode: str = "zscore",
    clip: Optional[Union[int, float]] = None,
) -> mne.time_frequency.AverageTFR:
    """Return power averaged over given channel types or picks."""
    if not isinstance(powers, list):
        powers = [powers]
    if isinstance(baseline, np.ndarray):
        baseline = [baseline]
    if isinstance(baseline, list):
        if not len(baseline) == len(powers):
            raise ValueError(
                "If numpy arrays are provided for"
                "c ustom baseline correction, the same number of numpy arrays"
                " and `power` (TFR) objects must be provided. Got:"
                f" {len(baseline)} baseline arrays and {len(powers)} powers."
            )

    power_all = None
    power_all_files = []
    for power in powers:
        power = power.copy().pick(picks=picks)
        if baseline:
            if not isinstance(baseline, list):
                _baseline = baseline
            else:
                _baseline = baseline
            power = apply_baseline(
                power=power, baseline=_baseline, baseline_mode=baseline_mode
            )  # type: ignore
        df_power = power.to_data_frame(picks=picks)
        freqs = power.freqs  # type: ignore
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
) -> Optional[
    Union[mne.time_frequency.AverageTFR, mne.time_frequency.EpochsTFR]
]:
    """Calculate power from single file."""
    print(f"File: {file.basename}")
    raw = mne_bids.read_raw_bids(file, verbose=False)

    try:
        if kwargs_preprocess:
            raw = pte.processing.preprocess(
                raw=raw,
                nm_channels_dir=nm_channels_dir,
                pick_used_channels=True,
                **kwargs_preprocess,
            )
        else:
            raw = pte.processing.preprocess(
                raw=raw,
                pick_used_channels=True,
                nm_channels_dir=nm_channels_dir,
            )
    except ValueError as error:
        if "No valid channels found" not in str(error):
            raise
        print(error)
        return None

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

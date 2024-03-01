"""Module for processing channels in electrophysiological data."""

from collections.abc import Sequence

import mne
import numpy as np

import pte.preprocessing.emg


def add_emg_rms(
    raw: mne.io.BaseRaw,
    ch_name: str,
    window_duration: int | float = 100,
    new_ch_name: str = "auto",
    analog_channel: str | Sequence[str] | None = None,
) -> mne.io.BaseRaw:
    """Add root mean square (RMS) of given bipolar EMG channel.

    Parameters
    ----------
    raw : MNE Raw object
        The MNE Raw object for this function to modify.
    ch_name : str
        Name of the bipolar EMG channel to be processed.
    window_duration : int | float. Default: 100
        Duration of the sliding RMS window in milliseconds.
    new_ch_name : str. Default: "auto"
        New name of the EMG RMS channel to be added.
    analog_channel : str | list of str | None. Default: None
        Optional names of channels with movement trace to be plotted.

    Returns
    -------
    The Raw object containing the added squared channel. Is a copy of the
    original Raw object.
    """
    if not raw.preload:
        raw.load_data()

    raw_rms = pte.preprocessing.emg.get_emg_rms(
        raw=raw.copy(),
        emg_ch=ch_name,
        window_duration=[window_duration],
        analog_channel=analog_channel,
        rereference=False,
        scale=1e-6,
    )

    rms_channel = f"EMG_RMS_{window_duration}"

    raw_rms.plot(
        scalings="auto", block=True, title="EMG Root Mean Square (RMS)"
    )

    raw = raw.add_channels(  # type: ignore
        [raw_rms.pick([rms_channel])],
        force_update_info=True,
    )
    if new_ch_name == "auto":
        side = "L" if "L" in ch_name else "R"
        new_ch_name = "_".join(
            ("EMG", side, "BR", "RMS", str(window_duration))
        )
    raw._orig_units[rms_channel] = "V"  # pylint: disable=protected-access
    raw = raw.rename_channels({rms_channel: new_ch_name})

    return raw


def add_squared_channel(
    raw: mne.io.BaseRaw,
    event_id: dict,
    ch_name: str,
    inplace: bool = False,
    scale: float = 1,
) -> mne.io.BaseRaw:
    """Create squared data (0s and 1s) from events and add to Raw object.

    Parameters
    ----------
    raw : MNE Raw object
        The MNE Raw object for this function to modify.
    event_id : dict | callable() | None | ‘auto’
        event_id (see MNE documentation) 'defining the annotations to be chosen
        from your Raw object. ONLY pass annotation names that should be used to
        generate the squared data'.
    ch_name : str
        Name for the squared channel to be added.
    inplace : bool. Default: False
        Set to True if Raw object should be modified in place.

    Returns
    -------
    raw : MNE Raw object
        The Raw object containing the added squared channel.
    """
    events, event_id = mne.events_from_annotations(raw, event_id)
    events_ids = events[:, 0]
    data_squared = np.zeros((1, raw.n_times))
    for i in np.arange(0, len(events_ids), 2):
        data_squared[0, events_ids[i] : events_ids[i + 1]] = 1.0 * scale

    info = mne.create_info(
        ch_names=[ch_name], ch_types=["misc"], sfreq=raw.info["sfreq"]
    )
    raw_squared = mne.io.RawArray(data_squared, info)
    raw_squared.set_meas_date(raw.info["meas_date"])
    raw_squared.info["line_freq"] = 50

    if not inplace:
        raw = raw.copy()
    if not raw.preload:
        raw.load_data()
    raw.add_channels([raw_squared], force_update_info=True)
    return raw


def summation_channel_name(summation_channels: Sequence[str]) -> str:
    """Create channel name for summation montage from given channels."""
    base_items: list[str] = []
    channel_numbers = []
    for ch_name in summation_channels:
        items = ch_name.split("_")
        channel_numbers.append(items.pop(2))
        if not base_items:
            base_items = items
    channel_number = f"({'+'.join(channel_numbers)})"
    base_items.insert(2, channel_number)
    summation_channel = "_".join(base_items)
    return summation_channel


def bipolar_channel_name(channels: Sequence[str]) -> str:
    """Create channel name for bipolar montage from two given channels."""
    if len(channels) != 2:
        raise ValueError(
            "Length of `channels` must be 2. Got:" f"{len(channels)}."
        )
    base_items: list[str] = []
    channel_numbers = []
    for ch_name in channels:
        items: list[str] = ch_name.split("_")
        channel_numbers.append(items.pop(2))
        if not base_items:
            base_items = items
    channel_number = f"{'-'.join(channel_numbers)}"
    base_items.insert(2, channel_number)
    new_channel = "_".join(base_items)
    return new_channel


def add_summation_channel(
    raw: mne.io.BaseRaw,
    summation_channels: Sequence[str],
    new_channel_name: str = "auto",
    inplace: bool = False,
    scale_data_by_factor: int | float | None = None,
    sort_channels: bool = True,
) -> mne.io.BaseRaw:
    """Sum up signals from given channels and add to MNE Raw object.

    Parameters
    ----------
    raw: MNE Raw object
        MNE Raw object containing data.
    summation_channels: list of str
        Channel names to be summed up.
    new_channel_name: str
        Channel name of new channel to be added
    inplace : bool. Default: False
        Set to True if Raw object should be modified in place.
    scale_data_by_factor
        Factor by which data to scale
    sort_channels: bool. Default: True
        Set to False if channel names should not be sorted alphabetically.

    Returns
    -------
    raw : MNE Raw object
        The Raw object containing the added squared channel.
    """
    if new_channel_name == "auto":
        new_channel_name = summation_channel_name(summation_channels)
    data = raw.get_data(picks=summation_channels)
    new_data = np.expand_dims(data.sum(axis=0), axis=0)
    if scale_data_by_factor is not None:
        new_data *= scale_data_by_factor
    ch_type = raw.get_channel_types(picks=summation_channels[0])
    info = mne.create_info(
        ch_names=[new_channel_name],
        sfreq=raw.info["sfreq"],
        ch_types=ch_type,
        verbose=False,
    )
    raw_new = mne.io.RawArray(
        new_data, info, first_samp=0, copy="auto", verbose=False
    )
    if not inplace:
        raw = raw.copy()
    if not raw.preload:
        raw.load_data()
    raw = raw.add_channels([raw_new], force_update_info=True)
    if sort_channels:
        raw.reorder_channels(sorted(raw.ch_names))
    return raw

"""Module for processing channels in electrophysiological data."""

from typing import Optional, Union

import mne
import numpy as np

import pte.preprocessing.emg


def add_emg_rms(
    raw: mne.io.BaseRaw,
    ch_name: str,
    window_duration: Union[int, float] = 100,
    new_ch_name: str = "auto",
    analog_channel: Optional[Union[str, list[str]]] = None,
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
        raw=raw,
        emg_ch=ch_name,
        window_duration=[window_duration],
        analog_ch=analog_channel,
        rereference=False,
    )

    rms_channel = f"EMG_RMS_{window_duration}"

    raw_rms.plot(scalings="auto", block=True, title="EMG Root Mean Square:")

    raw = raw.add_channels(  # type: ignore
        [raw_rms.pick_channels([rms_channel])],
        force_update_info=True,
    )
    if new_ch_name == "auto":
        side = "L" if "L" in ch_name else "R"
        new_ch_name = "_".join(
            ("EMG", side, "BR", "RMS", str(window_duration))
        )
    raw._orig_units[rms_channel] = "µV"
    raw = raw.rename_channels({rms_channel: new_ch_name})

    return raw


def add_squared_channel(
    raw: mne.io.BaseRaw, event_id: dict, ch_name: str, inplace: bool = False
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
    data = raw.get_data()
    events_ids = events[:, 0]
    data_squared = np.zeros((1, data.shape[1]))
    for i in np.arange(0, len(events_ids), 2):
        data_squared[0, events_ids[i] : events_ids[i + 1]] = 1

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


def _summation_channel_name(summation_channels: list[str]) -> str:
    """Create channel name from given channels."""
    base_items = None
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


def add_summation_channel(
    raw: mne.io.BaseRaw,
    summation_channels: list[str],
    new_channel_name: str = "auto",
    inplace: bool = False,
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

    Returns
    -------
    raw : MNE Raw object
        The Raw object containing the added squared channel.
    """
    if new_channel_name == "auto":
        new_channel_name = _summation_channel_name(summation_channels)
    data = raw.get_data(picks=summation_channels)
    new_data = np.expand_dims(data.sum(axis=0), axis=0)
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
    return raw

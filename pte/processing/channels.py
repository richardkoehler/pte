"""Module for processing channels in electrophysiological data."""

from typing import List
import mne
import numpy as np


def add_squared_channel(
    raw: mne.io.Raw, event_id: dict, ch_name: str
) -> mne.io.Raw:
    """Create squared data (0s and 1s) from events and add to Raw object.

    Parameters
    ----------
    raw : MNE Raw object
        The MNE Raw object for this function to modify.
    event_id : dict | callable() | None | ‘auto’
        event_id (see MNE documentation) defining the annotations to be chosen
        from your Raw object. ONLY pass annotation names that should be used to
        generate the squared data.
        Can be:
            dict: map descriptions (keys) to integer event codes (values). Only
            the descriptions present will be mapped, others will be ignored.
            callable: must take a string input and return an integer event code,
            or return None to ignore the event.
            None: Map descriptions to unique integer values based on their sorted order.
            ‘auto’ (default): prefer a raw-format-specific parser:
                Brainvision: map stimulus events to their integer part; response events
                to integer part + 1000; optic events to integer part + 2000;
                ‘SyncStatus/Sync On’ to 99998; ‘New Segment/’ to 99999; all others like
                None with an offset of 10000.
                Other raw formats: Behaves like None.
    ch_name : str
        Name for the squared channel to be added.

    Returns
    -------
    raw_final : MNE Raw object
        The Raw object containing the added squared channel. Is a copy of the
        original Raw object.
    """
    events, event_id = mne.events_from_annotations(raw, event_id)
    data = raw.get_data()
    evs_idx = events[:, 0]
    onoff = np.zeros((1, data.shape[1]))
    for i in np.arange(0, len(evs_idx), 2):
        onoff[0, evs_idx[i] : evs_idx[i + 1]] = 1
    info = mne.create_info(
        ch_names=[ch_name], ch_types=["misc"], sfreq=raw.info["sfreq"]
    )
    raw_sq = mne.io.RawArray(onoff, info)
    raw_sq.info.set_meas_date(raw.info["meas_date"])
    raw_sq.info["line_freq"] = 50
    raw_final = (
        raw.copy().load_data().add_channels([raw_sq], force_update_info=True)
    )
    return raw_final


def add_summation_channel(
    raw: mne.io.Raw, summation_channels: List[str], new_channel: str
) -> mne.io.Raw:
    """Sum up signals from given channels and add to MNE Raw object.

    Parameters
    ----------
    raw: MNE Raw object
        MNE Raw object containing data.
    summation_channels: list of str
        Channel names to be summed up.
    new_channel: str
        Channel name of new channel to be added

    Returns
    -------
    raw_final : MNE Raw object
        The Raw object containing the added channel. Is a copy of the
        original Raw object.
    """
    data = raw.get_data(picks=summation_channels)
    new_data = np.expand_dims(data.sum(axis=0), axis=0)
    ch_type = raw.get_channel_types(picks=summation_channels[0])
    info = mne.create_info(
        ch_names=[new_channel],
        sfreq=raw.info["sfreq"],
        ch_types=ch_type,
        verbose=False,
    )
    raw_new = mne.io.RawArray(
        new_data, info, first_samp=0, copy="auto", verbose=False
    )
    raw_final = (
        raw.copy().load_data().add_channels([raw_new], force_update_info=True)
    )
    return raw_final

"""Module for processing of EMG channels."""

from typing import Iterable, Union

import mne
import numpy as np
from numba import njit


def get_emg_rms(
    raw: mne.io.Raw,
    emg_ch: Union[str, list[str], np.ndarray],
    window_len: Union[float, int, Iterable],
    analog_ch: Union[list, str],
    rereference: bool = False,
    notch_filter: Union[float, int] = 50,
) -> mne.io.Raw:
    """Return root mean square with given window length of raw object.

    Parameters
    ----------
    raw : MNE raw object
        The data to be processed.
    emg_ch : list of str
        The EMG channels to be processed. Must be of length 1 or 2.
    window_len : float | int | array-like of float/int
        Window length(s) for root mean square calculation in milliseconds.
    analog_ch : str | list of str
        The target channel (e.g., rotameter) to be added to output raw object.
    rereference : boolean (optional)
        Set to True if EMG channels should be referenced in a bipolar montage.
        Default is False.

    Returns
    -------
    raw_rms : MNE raw object
        Raw object containing root mean square of windowed signal and target
        channel.
    """

    raw_emg = raw.copy().pick(picks=emg_ch).load_data(verbose=False)
    raw_emg.set_channel_types(
        mapping={name: "eeg" for name in raw_emg.ch_names}
    )
    if rereference:
        raw_emg = mne.set_bipolar_reference(
            inst=raw_emg,
            anode=raw_emg.ch_names[0],
            cathode=raw_emg.ch_names[1],
            ch_name=["EMG_BIP"],
            drop_refs=True,
            copy=False,
            verbose=False,
        )
    raw_emg.set_channel_types(
        mapping={name: "emg" for name in raw_emg.ch_names}
    )
    data_bip = raw_emg.get_data(verbose=False)[0]
    if notch_filter:
        assert isinstance(notch_filter, (int, float))
        freqs = np.arange(notch_filter, raw.info["sfreq"] / 2, notch_filter)
        raw_emg = raw_emg.notch_filter(freqs=freqs, picks="emg", verbose=False)
    raw_filt = raw_emg.filter(
        l_freq=15, h_freq=500, picks="all", verbose=False
    )
    if isinstance(window_len, (int, float)):
        window_len = [window_len]
    data = raw_filt.get_data()[0]
    data_arr = np.empty((len(window_len), len(data)))
    for idx, window in enumerate(window_len):
        data_rms = _rms_window_nb(data, window, raw.info["sfreq"])
        data_rms_zx = (data_rms - np.mean(data_rms)) / np.std(data_rms)
        data_arr[idx, :] = data_rms_zx
    data_analog = raw.copy().pick(picks=analog_ch).get_data()[0]
    if np.abs(min(data_analog)) > max(data_analog):
        data_analog = data_analog * -1
    data_all = np.vstack((data_analog, data_bip, data_arr))
    emg_ch_names = ["EMG_RMS_" + str(window) for window in window_len]
    info_rms = mne.create_info(
        ch_names=[analog_ch] + ["EMG_BIP"] + emg_ch_names,
        ch_types="emg",
        sfreq=raw.info["sfreq"],
    )
    raw_rms = mne.io.RawArray(data=data_all, info=info_rms, verbose=False)
    raw_rms.info["meas_date"] = raw.info["meas_date"]
    raw_rms.info["line_freq"] = raw.info["line_freq"]
    raw_rms.set_annotations(raw.annotations)
    raw_rms.set_channel_types({analog_ch: "misc"})
    return raw_rms


@njit
def _rms_window_nb(
    data: np.ndarray, window_len: Union[float, int], sfreq: Union[float, int]
) -> np.ndarray:
    """Return root mean square of input signal with given window length.

    Parameters
    ----------
    data : numpy.ndarray
        The data to be processed. Must be 1-dimensional.
    window_len : float | int
        Window length in milliseconds.
    sfreq : float | int
        Sampling frequency in 1/seconds.

    Returns
    -------
    data_rms
        Root mean square of windowed signal. Same dimension as input signal
    """

    half_window_size = int(sfreq * window_len / 1000 / 2)
    data_rms = np.empty_like(data)
    for i, dat_ in enumerate(data):
        if i in (0, len(data) - 1):
            data_rms[i] = np.absolute(dat_)
        elif i < half_window_size:
            new_window_size = i
            data_rms[i] = np.sqrt(
                np.mean(
                    np.power(
                        data[i - new_window_size : i + new_window_size], 2
                    )
                )
            )
        elif len(data) - i < half_window_size:
            new_window_size = len(data) - i
            data_rms[i] = np.sqrt(
                np.mean(
                    np.power(
                        data[i - new_window_size : i + new_window_size], 2
                    )
                )
            )
        else:
            data_rms[i] = np.sqrt(
                np.mean(
                    np.power(
                        data[i - half_window_size : i + half_window_size], 2
                    )
                )
            )
    return data_rms

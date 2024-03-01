"""Module for processing of EMG channels."""

from collections.abc import Sequence

import mne
import numpy as np
from numba import njit


def get_emg_rms(
    raw: mne.io.BaseRaw,
    emg_ch: str | list[str] | np.ndarray,
    window_duration: float | int | Sequence,
    analog_channel: str | Sequence[str] | None = None,
    rereference: bool = False,
    notch_filter: float | int = 50,
    scale: float = 1,
) -> mne.io.BaseRaw:
    """Return root mean square with given window length of raw object.

    Parameters
    ----------
    raw : MNE raw object
        The data to be processed.
    emg_ch : list of str
        The EMG channels to be processed. Must be of length 1 or 2.
    window_duration : float | int | array-like of float/int
        Window duration(s) for root mean square calculation in milliseconds.
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
    raw_emg: mne.io.BaseRaw = raw.copy()
    raw_emg.pick(picks=emg_ch)

    if not raw_emg.preload:
        print("Loading data")
        raw_emg.load_data()

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
        assert isinstance(notch_filter, int | float)
        freqs = np.arange(notch_filter, raw.info["sfreq"] / 2, notch_filter)
        raw_emg = raw_emg.notch_filter(freqs=freqs, picks="emg", verbose=False)
    raw_filtered = raw_emg.filter(
        l_freq=15, h_freq=500, picks="all", verbose=False
    )
    if isinstance(window_duration, int | float):
        window_duration = [window_duration]

    data = raw_filtered.get_data()[0]
    data_rms = np.empty((len(window_duration), len(data)))

    for idx, window in enumerate(window_duration):
        rms_raw = _rms_window_nb(data, window, raw.info["sfreq"])
        # now scale to match other MISC channels
        rms_zscore = (rms_raw - np.mean(rms_raw)) / np.std(rms_raw) * scale
        data_rms[idx, :] = rms_zscore

    emg_ch_names = [f"EMG_RMS_{window}" for window in window_duration]

    if analog_channel:
        if isinstance(analog_channel, str):
            analog_channel = [analog_channel]
        elif not isinstance(analog_channel, list):
            analog_channel = list(analog_channel)
        data_analog = raw.copy().pick(picks=analog_channel).get_data()[0]
        if np.abs(min(data_analog)) > max(data_analog):
            data_analog = data_analog * -1
        data_all = np.vstack((data_analog, data_bip, data_rms))
        ch_names = analog_channel + ["EMG_BIP"] + emg_ch_names
    else:
        data_all = np.vstack((data_bip, data_rms))
        ch_names = ["EMG_BIP"] + emg_ch_names

    # Create new raw object
    info_rms = mne.create_info(
        ch_names=ch_names,
        ch_types="misc",
        sfreq=raw.info["sfreq"],
    )
    raw_rms = mne.io.RawArray(data=data_all, info=info_rms, verbose=False)
    raw_rms.set_meas_date(raw.info["meas_date"])
    raw_rms.info["line_freq"] = raw.info["line_freq"]
    raw_rms.set_annotations(raw.annotations)
    raw_rms.set_channel_types({"EMG_BIP": "emg"})
    return raw_rms


@njit
def _rms_window_nb(
    data: np.ndarray, window_len: float | int, sfreq: float | int
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

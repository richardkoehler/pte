"""Plotting functions based on plotly."""

from pathlib import Path
from typing import Union
import numpy as np
import pandas as pd
from plotly import express
import scipy.signal

import mne


def plotly_mne(
    mne_raw: mne.io.BaseRaw,
    file_name: Union[Path, str],
    time_slice: tuple = (),
    plot_title: str = None,
    decimate: bool = True,
    normalize: bool = True,
    detrend: str = "linear",
    padding: Union[int, float] = 2,
) -> None:
    """
    Creates (export) the (sliced) MNE raw signal as an HTML plotly plot.

    Arguments
    ---------
        mne_raw: MNE raw object (output of mne.io.read_raw_...)
        file_name: name (and directory) for the exported html file
        time_slice: tuple of `start` and `end` slice (seconds)
            example: `t_slice = (1, 5)` returns the 1s-5s slice
        plot_title: Plot title (default is None)
        decimate: down-sampling (decimating) the signal to 200Hz sampling
            rate (default and recommended value is True)
        normalize: dividing the signal by the root mean square value for
            normalization (default and recommended value is True)
        detrend: The type of detrending.
            If do_detrend == 'linear' (default), the result of a linear
            least-squares fit to data is subtracted from data.
            If do_detrend == 'constant', only the mean of data is subtracted.
            else, no detrending.
        padding: multiplication factor for spacing between signals on the
            y-axis. For highly variant data, use higher values. default is 2.
    """
    samp_freq = int(mne_raw.info["sfreq"])
    channels_array = np.array(mne_raw.info["ch_names"])
    if time_slice:
        signals_array, time_array = mne_raw[
            :, time_slice[0] * samp_freq : time_slice[1] * samp_freq
        ]
    else:
        signals_array, time_array = mne_raw[:, :]

    plotly_rawdata(
        time_array,
        signals_array,
        channels_array,
        samp_freq,
        file_name,
        plot_title=plot_title,
        decimate=decimate,
        normalize=normalize,
        detrend=detrend,
        padding=padding,
    )


def plotly_rawdata(
    time_array: np.ndarray,
    signals_array: np.ndarray,
    channels_array: np.ndarray,
    sfreq: Union[int, float],
    file_name: Union[str, Path],
    plot_title: str = None,
    decimate: bool = True,
    normalize: bool = True,
    detrend: str = "linear",
    padding: Union[int, float] = 2,
) -> None:
    """
    Creates (export) the signals as an HTML plotly plot.

    Arguments
    ---------
        time_array: numpy array of time stamps (seconds)
        signals_array: a 2D-array of signals with shape (#channels, #samples)
        channels_array: numpy array (or list) of channel names
        samp_freq: sampling frequency (Hz)
        file_name: name (and directory) for the exported html file
        plot_title: Plot title (default is None)
        decimate: down-sampling (decimating) the signal to 200Hz sampling
            rate (default and recommended value is True)
        normalize: dividing the signal by the root mean square value for
            normalization (default and recommended value is True)
        detrend: The type of detrending.
            If do_detrend == 'linear' (default), the result of a linear
            least-squares fit to data is subtracted from data.
            If do_detrend == 'constant', only the mean of data is subtracted.
            else, no detrending.
        padding: multiplication factor for spacing between signals on the
            y-axis. For highly variant data, use higher values. default is 2.
    """

    time_array = np.squeeze(time_array)
    signals_array = np.squeeze(signals_array)
    channels_array = np.squeeze(channels_array)
    if signals_array.ndim == 1:
        signals_array = signals_array.reshape(1, -1)

    assert (
        signals_array.shape[0] == channels_array.shape[0]
    ), "signals_array ! channels_array Dimension mismatch!"
    assert (
        signals_array.shape[1] == time_array.shape[0]
    ), "signals_array ! time_array Dimension mismatch!"

    if decimate:
        decimate_factor = min(10, int(sfreq / 200))
        signals_array = scipy.signal.decimate(signals_array, decimate_factor)
        time_array = scipy.signal.decimate(time_array, decimate_factor)
    if detrend in ("linear", "constant"):
        signals_array = scipy.signal.detrend(
            signals_array, axis=1, type=detrend, overwrite_data=True
        )
    if normalize:
        eps_ = np.finfo(float).eps
        signals_array = signals_array / (
            _rms(signals_array, axis=1).reshape(-1, 1) + eps_
        )

    offset_value = padding * _rms(signals_array)  # RMS value
    signals_array = signals_array + offset_value * (
        np.arange(len(channels_array)).reshape(-1, 1)
    )

    signals_df = pd.DataFrame(
        data=signals_array.T, index=time_array, columns=channels_array
    )

    fig = express.line(
        signals_df,
        x=signals_df.index,
        y=signals_df.columns,
        line_shape="spline",
        render_mode="svg",
        labels=dict(index="Time (s)", value="(a.u.)", variable="Channel"),
        title=plot_title,
    )
    fig.update_layout(
        yaxis=dict(
            tickmode="array",
            tickvals=offset_value * np.arange(len(channels_array)),
            ticktext=channels_array,
        )
    )

    fig.write_html(str(file_name) + ".html")


def _rms(data: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Return the Root Mean Square (RMS) value of data along the given axis.
    """
    if axis >= data.ndim:
        raise ValueError(f"No {axis=} for data with {data.ndim} dimensions!")

    if axis < 0:
        return np.sqrt(np.mean(np.square(data)))
    return np.sqrt(np.mean(np.square(data), axis=axis))

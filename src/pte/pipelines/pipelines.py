"""Module for predefined processing pipelines."""
from collections.abc import Sequence
from pathlib import Path

import mne
import mne_bids

import pte


def process_emg_rms(
    raw: mne.io.BaseRaw,
    emg_channels: str | Sequence[str],
    window_duration: int | float = 100,
    annotate_trials: bool = True,
    add_squared_channel: bool = True,
    out_path: mne_bids.BIDSPath | None = None,
) -> mne.io.BaseRaw:
    """Add EMG root mean square channels to Raw object and save."""
    if raw.filenames:
        prefix = f"File: {Path(raw.filenames[0]).name}. "
    else:
        prefix = ""

    raw.plot(
        scalings="auto",
        block=True,
        title=(
            f"{prefix}Please check for bad channels and"
            " close window for code to continue to run."
        ),
    )

    if isinstance(emg_channels, str):
        emg_channels = [emg_channels]

    for emg_channel in emg_channels:
        analog_channel = [ch for ch in raw.ch_names if "ANALOG" in ch]
        analog_channel = analog_channel[0] if analog_channel else None
        raw = pte.preprocessing.add_emg_rms(
            raw=raw,
            ch_name=emg_channel,
            window_duration=window_duration,
            new_ch_name="auto",
            analog_channel=analog_channel,
        )

    if annotate_trials:
        raw = pte.preprocessing.annotate_trials(
            raw=raw, keyword="EMG", inplace=True
        )
        if add_squared_channel:
            raw = pte.preprocessing.add_squared_channel(
                raw=raw,
                event_id={"EMG_onset": 1, "EMG_end": -1},
                ch_name="SQUARED_EMG",
                inplace=True,
            )

            raw.plot(
                scalings="auto",
                block=True,
                title="SQUARED_EMG channel added. Please check this channel.",
            )

    if out_path:
        raw = pte.filetools.rewrite_bids_file(
            raw=raw, bids_path=out_path, reorder_channels=True
        )
    return raw

"""Module with preprocessing functions (resampling, referencing, etc.)"""

from pathlib import Path
from typing import Optional, Union, Sequence

import mne
import mne_bids
import numpy as np
import pandas as pd


def pick_by_nm_channels(
    raw: mne.io.BaseRaw,
    nm_channels_dir: Union[Path, str],
    fname: mne_bids.BIDSPath,
) -> mne.io.BaseRaw:
    """Pick channels (``used`` and ``good``) according to *nm_channels.csv."""
    raw = raw.copy()
    basename = str(Path(fname).stem)
    fpath = Path(nm_channels_dir) / Path(basename + "_nm_channels.csv")
    nm_channels: pd.DataFrame = pd.read_csv(fpath, header=0)
    channel_picks = nm_channels[(nm_channels["used"] == 1)]
    if len(channel_picks) == 0:
        raise ValueError(
            "No valid channels found in given nm_channels.csv file:"
            f" {fpath.name}"
        )
    return raw.pick_channels(ch_names=channel_picks["new_name"].to_list())


def bipolar_refs_from_nm_channels(
    nm_channels_dir: Path | str,
    filename: Path | str | mne_bids.BIDSPath,
    types: str | Sequence = ("ecog", "dbs", "eeg"),
) -> tuple[list, list, list]:
    """Get referencing montage from *nm_channels.csv file."""
    if not isinstance(types, Sequence):
        types = [types]
    if isinstance(filename, mne_bids.BIDSPath):
        # Implement this later
        # basename = filename.copy().update(
        # extension=None, suffix=None, datatype=None
        # ).basename
        basename = filename.copy().update(extension=None).basename
    else:
        basename = str(Path(filename).stem)
    fpath = Path(nm_channels_dir) / Path(basename + "_nm_channels.csv")
    nm_channels: pd.DataFrame = pd.read_csv(fpath, header=0)
    anodes, cathodes, ch_names = [], [], []
    for ch_type in types:
        df_picks = nm_channels.loc[
            (nm_channels.type == ch_type)
            & (nm_channels.used == 1)
            & (nm_channels.rereference != "average")
        ]
        anodes.extend(df_picks.name)
        cathodes.extend(df_picks.rereference)
        ch_names.extend(df_picks.new_name)
    return anodes, cathodes, ch_names


def bandstop_filter(
    raw: mne.io.BaseRaw,
    bandstop_freq: Optional[Union[str, int, float, np.ndarray]] = "auto",
    fname: Optional[str] = None,
) -> mne.io.BaseRaw:
    """Bandstop filter Raw data"""
    if bandstop_freq is None:
        return raw

    if isinstance(bandstop_freq, str):
        if bandstop_freq != "auto":
            raise ValueError(
                "`bandstop_freq` must be one of either `string`"
                f"`float`, `'auto'` or `None`. Got: {bandstop_freq}."
            )
        if not fname:
            try:
                fname = raw.filenames[0]
            except ValueError as error:
                raise ValueError(
                    "If `bandstop_freq` is `'auto'`, `fname` must be provided."
                ) from error
        if "StimOn" not in fname:
            return raw
        bandstop_freq = 130

    if isinstance(bandstop_freq, (int, float)):
        bandstop_freq = np.arange(
            bandstop_freq, raw.info["sfreq"] / 2, bandstop_freq
        )

    if bandstop_freq:
        print("FREQUENCIES:", bandstop_freq)
        raw = raw.notch_filter(
            bandstop_freq, notch_widths=bandstop_freq * 0.2, verbose=True
        )

    return raw


def preprocess(
    raw: mne.io.BaseRaw,
    nm_channels_dir: Path,
    filename: Optional[Union[str, Path, mne_bids.BIDSPath]] = None,
    average_ref_types: Sequence[str] | None = ("ecog",),
    line_freq: Optional[int] = None,
    resample_freq: Optional[Union[int, float]] = 500,
    high_pass: Optional[Union[int, float]] = None,
    low_pass: Optional[Union[int, float]] = None,
    bandstop_freq: Optional[Union[str, int, float, np.ndarray]] = "auto",
    pick_used_channels: bool = False,
) -> Optional[mne.io.BaseRaw]:
    """Preprocess data"""
    if line_freq is None:
        line_freq = raw.info["line_freq"]
    if filename is None:
        filename = raw.filenames[0]
    if isinstance(filename, Path):
        filename = str(filename)

    raw = raw.pick(picks=["ecog", "dbs"], verbose=False)
    if not raw.preload:
        raw.load_data(verbose=False)

    if average_ref_types:
        for pick_type in average_ref_types:
            raw = raw.set_eeg_reference(
                ref_channels="average", ch_type=pick_type, verbose=False
            )
            raw = raw.rename_channels(
                {
                    ch: f"{ch}-avgref"
                    for ch, ch_type in zip(
                        raw.ch_names, raw.get_channel_types()
                    )
                    if ch_type == pick_type
                }
            )

    anodes, cathodes, ch_names = bipolar_refs_from_nm_channels(
        nm_channels_dir=nm_channels_dir, filename=filename
    )
    if not ch_names:
        print("No channels given for bipolar re-referencing.")
    else:
        raw = mne.set_bipolar_reference(
            raw,
            anode=anodes,
            cathode=cathodes,
            ch_name=ch_names,
            drop_refs=True,
        )

    raw = raw.resample(sfreq=resample_freq, verbose=False)

    raw = raw.filter(l_freq=high_pass, h_freq=low_pass, verbose=False)

    notch_freqs = np.arange(line_freq, raw.info["sfreq"] / 2, line_freq)
    if notch_freqs.size > 0:
        raw = raw.notch_filter(notch_freqs, verbose=False)

    raw = bandstop_filter(raw=raw, bandstop_freq=bandstop_freq)

    if pick_used_channels:
        raw = pick_by_nm_channels(
            raw=raw, nm_channels_dir=nm_channels_dir, fname=filename
        )

    raw = raw.reorder_channels(sorted(raw.ch_names))
    return raw

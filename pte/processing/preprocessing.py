"""Module with preprocessing functions (resampling, referencing, etc.)"""

from pathlib import Path
from typing import Optional, Union

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
    channel_picks = nm_channels[
        (nm_channels["used"] == 1) & (nm_channels["status"] == "good")
    ]
    if len(channel_picks) == 0:
        raise ValueError(
            "No valid channels found in given nm_channels.csv file:"
            f" {fpath.name}"
        )
    return raw.pick_channels(ch_names=channel_picks["name"].to_list())


def references_from_nm_channels(
    nm_channels_dir: Union[Path, str],
    fname: Union[Path, str, mne_bids.BIDSPath],
    types: Union[str, list] = "dbs",
) -> tuple[list, list, list]:
    """Get referencing montage from *nm_channels.csv file."""
    if not isinstance(types, list):
        types = [types]
    basename = str(Path(fname).stem)
    fpath = Path(nm_channels_dir) / Path(basename + "_nm_channels.csv")
    nm_channels: pd.DataFrame = pd.read_csv(fpath, header=0)
    anodes, cathodes, ch_names = [], [], []
    for ch_type in types:
        df_picks = nm_channels.loc[
            (nm_channels.type == ch_type) & (nm_channels.used == 1)
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
                fnames = raw.filenames
                fname = raw.filenames[0]
            except ValueError:
                raise ValueError(
                    "If `bandstop_freq` is `'auto'`, `fname` must be provided."
                )
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
    line_freq: Optional[int] = None,
    fname: Optional[Union[str, Path]] = None,
    resample_freq: Optional[Union[int, float]] = 500,
    bandstop_freq: Optional[Union[str, int, float, np.ndarray]] = "auto",
    pick_used_channels: bool = False,
) -> Optional[mne.io.BaseRaw]:
    """Preprocess data"""
    if not line_freq:
        line_freq = raw.info["line_freq"]
    if fname is None:
        fname = raw.filenames[0]
    elif isinstance(fname, Path):
        fname = str(fname)

    raw = raw.pick(picks=["ecog", "dbs"], verbose=False)
    raw = raw.load_data(verbose=False)
    raw = raw.resample(sfreq=resample_freq, verbose=False)
    notch_freqs = np.arange(line_freq, raw.info["sfreq"] / 2, line_freq)
    raw = raw.notch_filter(notch_freqs, verbose=False)

    raw = bandstop_filter(raw=raw, bandstop_freq=bandstop_freq)

    raw = raw.set_eeg_reference(
        ref_channels="average", ch_type="ecog", verbose=False
    )
    anodes, cathodes, ch_names = references_from_nm_channels(
        nm_channels_dir=nm_channels_dir, fname=fname
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

    if pick_used_channels:
        raw = pick_by_nm_channels(
            raw=raw, nm_channels_dir=nm_channels_dir, fname=fname
        )

    raw = raw.reorder_channels(sorted(raw.ch_names))
    return raw

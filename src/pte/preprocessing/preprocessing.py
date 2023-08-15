"""Module for preprocessing functions (resampling, referencing, etc.)"""
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import mne
import mne_bids
import numpy as np
import pandas as pd


def pick_by_nm_channels(
    raw: mne.io.BaseRaw,
    nm_channels_dir: Path | str,
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
    raw.pick(channel_picks["new_name"].to_list())
    return raw


def bipolar_refs_from_nm_channels(
    nm_channels_dir: Path | str,
    filename: Path | str | mne_bids.BIDSPath,
    types: str | Sequence = ("ecog", "dbs", "eeg"),
) -> tuple[list[str], list[str], list[str]]:
    """Get referencing montage from *nm_channels.csv file."""
    if not isinstance(types, Sequence):
        types = (types,)
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
            & nm_channels.rereference.notna()
            & (nm_channels.rereference != "None")
            & (nm_channels.rereference != "average")
        ]
        anodes.extend(df_picks.name)
        cathodes.extend(df_picks.rereference)
        ch_names.extend(df_picks.new_name)
    return anodes, cathodes, ch_names


def bandstop_filter(
    raw: mne.io.BaseRaw,
    bandstop_freq: str | int | float | np.ndarray | None = "auto",
    fname: str | None = None,
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
        if not isinstance(fname, str):
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
    filename: Path | str | mne_bids.BIDSPath | None = None,
    average_ref_types: Sequence[str] | str | None = None,
    ref_nm_channels: bool = True,
    notch_filter: int | Literal["auto"] | None = "auto",
    resample_freq: int | float | None = 500,
    high_pass: int | float | None = None,
    low_pass: int | float | None = None,
    bandstop_freq: str | int | float | np.ndarray | None = "auto",
    pick_used_channels: bool = False,
) -> mne.io.BaseRaw:
    """Preprocess raw data."""
    if notch_filter == "auto":
        notch_filter = raw.info["line_freq"]
    if filename is None:
        filename = raw.filenames[0]
    if isinstance(filename, Path):
        filename = str(filename)

    # raw.pick(picks=["ecog", "dbs"], verbose=False)
    if not raw.preload:
        raw.load_data(verbose=True)

    if average_ref_types:
        if isinstance(average_ref_types, str):
            average_ref_types = [average_ref_types]
        for pick_type in average_ref_types:
            raw.set_eeg_reference(
                ref_channels="average", ch_type=pick_type, verbose=True
            )
            raw.rename_channels(
                {
                    ch: f"{ch}-avgref"
                    for ch, ch_type in zip(
                        raw.ch_names, raw.get_channel_types()
                    )
                    if ch_type == pick_type
                }
            )

    if ref_nm_channels:
        anodes, cathodes, ch_names = bipolar_refs_from_nm_channels(
            nm_channels_dir=nm_channels_dir, filename=filename
        )
        if not ch_names:
            print("No channels given for bipolar re-referencing.")
        else:
            # Renaming necessary to account for possible name duplications
            # anodes_map = {anode: f"{anode}_old" for anode in anodes}
            # raw.rename_channels(anodes_map)
            raw = mne.set_bipolar_reference(  # type: ignore
                raw,
                anode=anodes,  # list(anodes_map.values()),
                cathode=cathodes,
                ch_name=ch_names,
                drop_refs=True,
            )
            bads = raw.info["bads"]
            for ch in ch_names:
                if ch in bads:
                    bads.remove(ch)

    if resample_freq is not None:
        raw.resample(sfreq=resample_freq, verbose=True)

    if high_pass is not None or low_pass is not None:
        raw.filter(l_freq=high_pass, h_freq=low_pass, verbose=True)

    if notch_filter is not None:
        notch_freqs = np.arange(
            notch_filter, raw.info["sfreq"] / 2, notch_filter
        )
        if notch_freqs.size > 0:
            raw.notch_filter(notch_freqs, verbose=True)

    raw = bandstop_filter(raw=raw, bandstop_freq=bandstop_freq)

    if pick_used_channels:
        raw = pick_by_nm_channels(
            raw=raw, nm_channels_dir=nm_channels_dir, fname=filename
        )

    raw.reorder_channels(sorted(raw.ch_names))
    return raw

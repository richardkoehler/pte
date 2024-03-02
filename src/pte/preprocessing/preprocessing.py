"""Module for preprocessing functions (resampling, referencing, etc.)"""

from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import mne
import mne_bids
import numpy as np
import pandas as pd


def pick_by_nm_channels(
    raw: mne.io.BaseRaw, nm_channels: pd.DataFrame
) -> mne.io.BaseRaw:
    """Pick channels (``used`` == 1) according to *nm_channels.csv."""
    channel_picks = nm_channels[(nm_channels["used"] == 1)]
    if len(channel_picks) == 0:
        raise ValueError(
            "No valid channels found in given nm_channels DataFrame."
            "Please check the `used` column."
        )
    raw.pick(channel_picks["new_name"].to_list())
    return raw


def load_nm_channels(
    nm_channels_dir: Path | str,
    filename: Path | str | mne_bids.BIDSPath,
) -> pd.DataFrame:
    """Load *nm_channels.csv file."""
    if isinstance(filename, Path):
        basename = filename.stem
    elif isinstance(filename, mne_bids.BIDSPath):
        basename = filename.copy().update(extension=None).basename
    else:
        basename = Path(filename).stem
    fpath = Path(nm_channels_dir, f"{basename}_nm_channels.csv")
    nm_channels: pd.DataFrame = pd.read_csv(fpath, header=0)
    return nm_channels


def stim_filter(
    raw: mne.io.BaseRaw,
    stim_freq: str | int | float | np.ndarray | None = "auto",
    fname: str | None = None,
    verbose: bool | str | int | None = False,
) -> mne.io.BaseRaw:
    """Bandstop filter Raw data"""
    if stim_freq is None:
        return raw

    if isinstance(stim_freq, str):
        if stim_freq != "auto":
            raise ValueError(
                "`stim_freq` must be one of either `int`"
                f"`float`, `'auto'` or `None`. Got: {stim_freq}."
            )
        if not isinstance(fname, str):
            try:
                fname = raw.filenames[0]
            except ValueError as error:
                raise ValueError(
                    "If `stim_freq` is `'auto'`, `fname` must be provided."
                ) from error
        if "StimOn" not in fname:
            return raw
        stim_freq = 130

    if isinstance(stim_freq, int | float):
        stim_freq = np.arange(stim_freq, raw.info["sfreq"] // 2 - 1, stim_freq)

    notch_widths = (stim_freq * 0.2).clip(max=26)
    raw = raw.notch_filter(
        stim_freq, notch_widths=notch_widths, verbose=verbose
    )
    return raw


def bipolar_refs_from_nm_channels(
    nm_channels,
) -> tuple[list[str], list[str], list[str]]:
    """Get referencing montage from nm_channels DataFrame."""
    anodes, cathodes, ch_names = [], [], []
    df_picks = nm_channels.loc[
        (nm_channels.used == 1)
        & nm_channels.rereference.notna()
        & (nm_channels.rereference != "None")
        & (nm_channels.rereference != "average")
    ]
    anodes.extend(df_picks.name)
    cathodes.extend(df_picks.rereference)
    ch_names.extend(df_picks.new_name)
    return anodes, cathodes, ch_names


def ref_by_nm_channels(
    raw: mne.io.BaseRaw, nm_channels: pd.DataFrame
) -> mne.io.BaseRaw:
    anodes, cathodes, new_names = bipolar_refs_from_nm_channels(nm_channels)
    if not new_names:
        print("No channels given for bipolar re-referencing.")
        return raw
    # Renaming necessary to account for possible name duplications
    curr_names = raw.ch_names
    rename = {}
    for i, ch in enumerate(new_names):
        if ch in curr_names:
            new_name = f"{ch}_new"
            new_names[i] = new_name
            rename[new_name] = ch
    raw = mne.set_bipolar_reference(  # type: ignore
        raw,
        anode=anodes,
        cathode=cathodes,
        ch_name=new_names,
        drop_refs=False,
    )

    if rename:
        raw.drop_channels(rename.values())
        raw.rename_channels(rename)

    drop = list(set(anodes + cathodes) & set(raw.ch_names))
    keep = nm_channels.query("used == 1 and name not in @anodes")
    if not keep.empty:
        drop = [ch for ch in drop if ch not in keep["name"].tolist()]
    if drop:
        raw.drop_channels(drop)

    bads = raw.info["bads"]
    for ch in new_names:
        if ch in bads:
            bads.remove(ch)
    return raw


def preprocess(
    raw: mne.io.BaseRaw,
    nm_channels_dir: Path | None = None,
    filename: Path | str | mne_bids.BIDSPath | None = None,
    average_ref_types: Sequence[str] | str | None = None,
    ref_nm_channels: bool = True,
    notch_filter: int | Literal["auto"] | None = "auto",
    resample_freq: int | float | None = 500,
    high_pass: int | float | None = None,
    low_pass: int | float | None = None,
    stim_freq: str | int | float | np.ndarray | None = "auto",
    pick_used_channels: bool = False,
    sort_channels: bool = True,
    verbose: bool | str | int | None = True,
) -> mne.io.BaseRaw:
    """Preprocess raw data."""
    if (pick_used_channels or ref_nm_channels) and nm_channels_dir is None:
        raise ValueError("`nm_channels_dir` must be provided.")
    if nm_channels_dir:
        if filename is None:
            if not raw.filenames:
                raise ValueError(
                    "If `filename` is not provided, `raw` must have a"
                    " `filenames` attribute."
                )
            filename = raw.filenames[0]
        nm_channels = load_nm_channels(nm_channels_dir, filename)

    if notch_filter == "auto":
        notch_filter = raw.info["line_freq"]

    if not raw.preload:
        raw.load_data(verbose=True)

    if average_ref_types:
        if isinstance(average_ref_types, str):
            average_ref_types = [average_ref_types]
        for pick_type in average_ref_types:
            raw.set_eeg_reference(
                ref_channels="average", ch_type=pick_type, verbose=verbose
            )
            raw.rename_channels(
                {
                    ch: f"{ch}-avgref"
                    for ch, ch_type in zip(
                        raw.ch_names, raw.get_channel_types(), strict=True
                    )
                    if ch_type == pick_type
                }
            )

    if ref_nm_channels and nm_channels_dir:
        raw = ref_by_nm_channels(raw=raw, nm_channels=nm_channels)

    if nm_channels_dir:
        names = nm_channels.query("used == 1 or target == 1")
        names = names[["name", "new_name"]].dropna()
        old_names = names["name"].to_list()
        new_names = names["new_name"].to_list()
        curr_names = raw.ch_names
        rename_map = {
            old: new
            for old, new in zip(old_names, new_names, strict=True)
            if old in curr_names
        }
        raw.rename_channels(rename_map)

    if pick_used_channels:
        raw = pick_by_nm_channels(raw=raw, nm_channels=nm_channels)

    if resample_freq is not None:
        raw.resample(sfreq=resample_freq, verbose=verbose)

    if high_pass is not None or low_pass is not None:
        raw.filter(l_freq=high_pass, h_freq=low_pass, verbose=verbose)

    if notch_filter is not None:
        notch_freqs: np.ndarray = np.arange(
            notch_filter, raw.info["sfreq"] / 2, notch_filter
        )
        if notch_freqs.size > 0:
            raw.notch_filter(notch_freqs, verbose=verbose)

    if stim_freq is not None:
        raw = stim_filter(raw=raw, stim_freq=stim_freq, verbose=verbose)

    if sort_channels:
        raw.reorder_channels(sorted(raw.ch_names))
    return raw

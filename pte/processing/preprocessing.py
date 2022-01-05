"""Module with preprocessing functions (resampling, referencing, etc.)"""

from pathlib import Path
from typing import Optional, Union

import mne
import mne_bids
import numpy as np
import pandas as pd


def references_from_nm_channels(
    nm_channels_dir: Union[Path, str],
    fname: mne_bids.BIDSPath,
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


def preprocess(
    raw: mne.io.BaseRaw,
    nm_channels_dir: Path,
    line_freq: Optional[int] = None,
    fname: Optional[Union[str, Path]] = None,
    resample_freq: Optional[Union[int, float]] = 500,
) -> mne.io.BaseRaw:
    """Preprocess data"""
    if not line_freq:
        line_freq = raw.info["line_freq"]
    if not fname:
        fname = raw.filenames[0]

    raw = raw.pick(picks=["ecog", "dbs"], verbose=False)
    raw = raw.load_data(verbose=False)
    raw = raw.resample(sfreq=resample_freq, verbose=False)
    notch_freqs = np.arange(line_freq, raw.info["sfreq"] / 2, line_freq)
    raw = raw.notch_filter(notch_freqs, verbose=False)
    raw = raw.set_eeg_reference(
        ref_channels="average", ch_type="ecog", verbose=False
    )
    anodes, cathodes, ch_names = references_from_nm_channels(
        nm_channels_dir=nm_channels_dir, fname=fname
    )
    raw = mne.set_bipolar_reference(
        raw, anode=anodes, cathode=cathodes, ch_name=ch_names, drop_refs=True,
    )
    raw = raw.reorder_channels(sorted(raw.ch_names))
    return raw

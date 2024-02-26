"""Module for handling event annotations."""

from pathlib import Path

import mne_bids
import pandas as pd


def get_bad_epochs(
    filename: Path | str | mne_bids.BIDSPath,
    bad_epochs_dir: Path | str,
) -> pd.DataFrame:
    """Get DataFrame of bad epochs from *_badepochs file."""
    if not isinstance(filename, mne_bids.BIDSPath):
        filename = mne_bids.get_bids_path_from_fname(Path(filename))
    basename = filename.copy().update(suffix=None, datatype=None).basename
    full_path = Path(bad_epochs_dir, f"{basename}_badepochs.csv")
    bad_epochs = pd.read_csv(
        full_path,
        dtype={"event_id": int, "event_description": str, "reason": str},
    )
    return bad_epochs

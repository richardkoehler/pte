"""Module for handling event annotations."""

from pathlib import Path
from typing import Optional, Union

import mne_bids
import pandas as pd


def get_bad_epochs(
    filename: Path | str | mne_bids.BIDSPath,
    bad_epochs_dir: Path | str | None = None,
) -> pd.DataFrame:
    """Get DataFrame of bad epochs from *_badepochs file."""
    if bad_epochs_dir is not None:
        if isinstance(filename, mne_bids.BIDSPath):
            basename = (
                filename.copy().update(suffix=None, datatype=None).basename
            )
        else:
            basename = Path(filename).stem[:-5]
        filename = Path(bad_epochs_dir, f"{basename}_badepochs.csv")
    bad_epochs = pd.read_csv(filename, index_col=0)
    return bad_epochs

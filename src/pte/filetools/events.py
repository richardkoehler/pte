"""Module for handling event annotations."""

from pathlib import Path
from typing import Optional, Union

import mne_bids
import numpy as np
import pandas as pd


def get_bad_events(
    bad_events_path: Optional[Union[Path, mne_bids.BIDSPath, str]],
    fname: Union[Path, str],
) -> np.ndarray:
    """Get DataFrame of bad events from bad events path."""
    if not bad_events_path:
        return np.atleast_1d([])
    bad_events_path = Path(bad_events_path)
    if bad_events_path.is_dir():
        basename = Path(fname).stem
        bad_events_path = bad_events_path / (basename + "_bad_epochs.csv")
    if not bad_events_path.exists():
        print(f"No bad epochs file found for: {str(fname)}")
        return np.atleast_1d([])
    bad_events = pd.read_csv(bad_events_path, index_col=0).event_id.to_numpy()
    return bad_events

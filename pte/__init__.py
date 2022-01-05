"""Package for electrophysiological analyses."""

from . import decoding, filetools, plotting, processing, stats, time_frequency
from .filetools import (
    add_coord_column,
    get_bids_electrodes,
    get_filefinder,
    loadmat,
    rewrite_bids_file,
)

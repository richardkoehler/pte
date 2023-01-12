"""Modules for handling and filtering files."""

from .bids import (
    add_coord_column,
    get_bids_electrodes,
    rewrite_bids_file,
    sub_med_stim_from_fname,
)
from .events import get_bad_epochs
from .filefinder import get_filefinder
from .matlab import loadmat

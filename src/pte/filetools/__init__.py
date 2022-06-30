"""Modules for handling and filtering files."""

from .bids import (
    add_coord_column,
    get_bids_electrodes,
    sub_med_stim_from_fname,
    rewrite_bids_file,
)
from .events import get_bad_epochs
from .filefinder import get_filefinder
from .matlab import loadmat

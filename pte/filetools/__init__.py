"""Modules for handling and filtering files."""

from .filefinder import get_filefinder
from .matlab import loadmat
from .bids import add_coord_column, rewrite_bids_file, get_bids_electrodes
from .events import get_bad_events

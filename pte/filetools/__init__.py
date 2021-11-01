"""Modules for handling and filtering files."""

from .filereader import get_filereader
from .matlab import loadmat
from .bids import add_coord_column, bids_rewrite_file, bids_get_coords

"""Add-on package for flexible machine learning using py_neuromodulation."""

from .filetools import (
    add_coord_column,
    get_filereader,
    loadmat,
    bids_rewrite_file,
    bids_get_coords,
)
from .plot import raw_plotly, sig_plotly

from .decode import classification
from . import filetools
from . import process

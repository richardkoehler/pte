"""Add-on package for flexible machine learning using py_neuromodulation."""

from .filetools import (
    add_coord_column,
    get_filefinder,
    loadmat,
    rewrite_bids_file,
    get_bids_coords,
)
from .plotting import raw_plotly, sig_plotly

from .decoding import run_experiment
from . import filetools
from . import processing
from . import stats

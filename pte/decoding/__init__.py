"""Modules for machine learning."""

from .decode import get_decoder
from .experiment import Experiment
from .experiment_factory import run_experiment
from .load import (
    load_predictions,
    load_results,
    load_results_singlechannel,
)
from .plot import boxplot_results, lineplot_prediction, violinplot_results
from .timepoint import get_earliest_timepoint

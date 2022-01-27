"""Define abstract base classes to construct Model classes."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Union

import numpy as np
from imblearn.over_sampling import (
    ADASYN,
    BorderlineSMOTE,
    RandomOverSampler,
    SMOTE,
)
from imblearn.under_sampling import RandomUnderSampler
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils.class_weight import compute_sample_weight


@dataclass
class Decoder(ABC):
    """Basic representation of class of machine learning decoders."""

    scoring: Any
    balancing: Optional[str] = "oversample"
    optimize: bool = False
    model: Any = field(init=False)
    data_train: pd.DataFrame = field(init=False)
    labels_train: np.ndarray = field(init=False)
    groups_train: Iterable = field(init=False)

    @abstractmethod
    def fit(self, data: pd.DataFrame, labels: np.ndarray, groups) -> None:
        """Fit model to given training data and training labels."""

    def get_score(self, data_test: np.ndarray, label_test: np.ndarray):
        """Calculate score."""
        return self.scoring(self.model, data_test, label_test)

    @staticmethod
    def _get_validation_split(
        data: pd.DataFrame,
        labels: np.ndarray,
        groups: np.ndarray,
        train_size: float = 0.8,
    ) -> tuple[Union[pd.DataFrame, pd.Series], np.ndarray, list]:
        """Split data into single training and validation set."""
        val_split = GroupShuffleSplit(n_splits=1, train_size=train_size)
        train_ind, val_ind = next(val_split.split(data, labels, groups))
        data_train, data_val = (
            data.iloc[train_ind],
            data.iloc[val_ind],
        )
        labels_train, labels_val = (
            labels[train_ind],
            labels[val_ind],
        )
        eval_set = [(data_val, labels_val)]
        return data_train, labels_train, eval_set

    @staticmethod
    def _balance_samples(
        data: np.ndarray, labels: np.ndarray, method: str = "oversample"
    ) -> tuple:
        """Balance class sizes to create equal class distributions.

        Parameters
        ----------
        data : numpy.ndarray of shape (n_features, n_samples)
            Data or features.
        labels : numpy.ndarray of shape (n_samples, )
            Array of class disribution
        method : {'oversample', 'undersample', 'weight'}
            Method to be used for rebalancing classes. 'oversample' will upsample
            the class with less samples. 'undersample' will downsample the class
            with more samples. 'weight' will generate balanced class weights.
            Default: 'oversample'

        Returns
        -------
        data : numpy.ndarray
            Rebalanced feature array of shape (n_features, n_samples)
        labels : numpy.ndarray
            Corresponding class distributions. Class sizes are now evenly balanced.
        sample_weight: numpy.ndarray of shape (n_samples, ) | None
            Sample weights if method = 'weight' else None
        """
        BALANCING_METHODS = [
            "oversample",
            "smote",
            "borderline_smote",
            "adasyn",
            "undersample",
            "balance_weights",
            True,
            False,
        ]
        sample_weight = None
        if np.mean(labels) != 0.5:
            if method == "oversample":
                ros = RandomOverSampler(sampling_strategy="auto")
                data, labels = ros.fit_resample(data, labels)
            elif method == "smote":
                ros = SMOTE(sampling_strategy="auto", k_neighbors=5)
                data, labels = ros.fit_resample(data, labels)
            elif method == "borderline_smote":
                ros = BorderlineSMOTE(
                    sampling_strategy="auto",
                    k_neighbors=5,
                    kind="borderline-1",
                )
                data, labels = ros.fit_resample(data, labels)
            elif method == "adasyn":
                try:
                    ros = ADASYN(sampling_strategy="auto", n_neighbors=5)
                    data, labels = ros.fit_resample(data, labels)
                except ValueError as error:
                    if (
                        len(error.args) > 0
                        and error.args[0]
                        == "No samples will be generated with the provided ratio settings."
                    ):
                        pass
                    else:
                        raise error
            elif method == "undersample":
                ros = RandomUnderSampler(sampling_strategy="auto")
                data, labels = ros.fit_resample(data, labels)
            elif method == "weight":
                sample_weight = compute_sample_weight(
                    class_weight="balanced", y=labels
                )
            else:
                raise BalancingMethodNotFoundError(method, BALANCING_METHODS)
        return data, labels, sample_weight


class BalancingMethodNotFoundError(Exception):
    """Exception raised when invalid balancing method is passed.

    Attributes:
        input_value -- input value which caused the error
        allowed -- allowed input values
        message -- explanation of the error
    """

    def __init__(
        self,
        input_value,
        allowed,
        message="Input balancing method is not an allowed value.",
    ) -> None:
        self.input_value = input_value
        self.allowed = allowed
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f"{{self.message}} Allowed values: {self.allowed}. Got: {self.input_value}."

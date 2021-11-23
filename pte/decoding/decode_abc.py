"""Define abstract base classes to construct Model classes."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

import numpy as np
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils.class_weight import compute_sample_weight


@dataclass
class Decoder(ABC):
    """Basic representation of class of machine learning decoders."""

    scoring: Any
    balancing: Optional[str] = "oversample"
    optimize: bool = False
    model: Any = field(init=False)
    data_train: np.ndarray = field(init=False)
    labels_train: np.ndarray = field(init=False)
    groups_train: Iterable = field(init=False)

    @abstractmethod
    def fit(self, data: np.ndarray, labels: np.ndarray, groups) -> None:
        """Fit model to given training data and training labels."""

    def get_score(self, data_test: np.ndarray, label_test: np.ndarray):
        """Calculate score."""
        return self.scoring(self.model, data_test, label_test)

    @staticmethod
    def _get_validation_split(
        data, labels, groups, train_size: float = 0.75
    ) -> tuple:
        """Split data into single training and validation set."""
        val_split = GroupShuffleSplit(n_splits=1, train_size=train_size)
        for train_ind, val_ind in val_split.split(data, labels, groups):
            data_train, data_val = (
                data[train_ind],
                data[val_ind],
            )
            labels_train, labels_val = (
                labels[train_ind],
                labels[val_ind],
            )
            eval_set = [(data_val, labels_val)]
        return data_train, labels_train, eval_set

    @staticmethod
    def _balance_samples(
        data: np.ndarray, target: np.ndarray, method: str = "oversample"
    ) -> tuple:
        """Balance class sizes to create equal class distributions.

        Parameters
        ----------
        data : numpy.ndarray of shape (n_features, n_samples)
            Data or features.
        target : numpy.ndarray of shape (n_samples, )
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
        target : numpy.ndarray
            Corresponding class distributions. Class sizes are now evenly balanced.
        sample_weight: numpy.ndarray of shape (n_samples, ) | None
            Sample weights if method = 'weight' else None
        """
        sample_weight = None
        if np.mean(target) != 0.5:
            if method == "oversample":
                ros = RandomOverSampler(sampling_strategy="auto")
                data, target = ros.fit_resample(data, target)
            elif method == "undersample":
                ros = RandomUnderSampler(sampling_strategy="auto")
                data, target = ros.fit_resample(data, target)
            elif method == "weight":
                sample_weight = compute_sample_weight(
                    class_weight="balanced", y=target
                )
            else:
                raise ValueError(
                    f"Method not identified. Given method was " f"{method}."
                )
        return data, target, sample_weight

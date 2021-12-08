"""Module for machine learning models."""

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from bayes_opt import BayesianOptimization
from catboost import CatBoostClassifier
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, log_loss
from sklearn.model_selection import GroupShuffleSplit
from sklearn.svm import SVC
from xgboost import XGBClassifier

from .decode_abc import Decoder


def get_decoder(
    classifier: str,
    scoring: str = "balanced_accuracy",
    balancing: Optional[str] = None,
    optimize: bool = False,
) -> Decoder:
    """Create and return Decoder of desired type.

    Parameters
    ----------
    classifier : str
        Allowed values for `classifier`: ["catboost", "lda", "lin_svm", "lr", "svm_lin",
        "svm_poly", "svm_rbf", "xgb"].
    scoring : str | None, default="balanced_accuracy"
        Score to be calculated. Possible values:
        ["oversample", "undersample", "balance_weights"].
    balancing : str | None, default=None
        Method for balancing skewed datasets. Possible values:
        ["oversample", "undersample", "balance_weights"].

    Returns
    -------
    Decoder
        Instance of Decoder given `classifer` and `balancing` method.
    """
    CLASSIFIERS = {
        "catboost": CATB,
        "dummy": Dummy,
        "lda": LDA,
        "lr": LR,
        "qda": QDA,
        "svm_lin": SVC_Lin,
        "svm_poly": SVC_Poly,
        "svm_rbf": SVC_RBF,
        "xgb": XGB,
    }
    SCORING_METHODS = {
        "balanced_accuracy": _get_balanced_accuracy,
        "log_loss": _get_log_loss,
    }

    classifier = classifier.lower()
    balancing = balancing.lower() if isinstance(balancing, str) else balancing
    scoring = scoring.lower()

    if classifier not in CLASSIFIERS:
        raise DecoderNotFoundError(classifier, CLASSIFIERS.keys())
    if scoring not in SCORING_METHODS:
        raise ScoringMethodNotFoundError(scoring, SCORING_METHODS.keys())
    return CLASSIFIERS[classifier](
        balancing=balancing,
        optimize=optimize,
        scoring=SCORING_METHODS[scoring],
    )


def _get_balanced_accuracy(model, data_test, label_test) -> Any:
    """Calculated balanced accuracy score."""
    return balanced_accuracy_score(label_test, model.predict(data_test))


def _get_log_loss(model, data_test, label_test) -> Any:
    """Calculate Log Loss score."""
    return log_loss(label_test, model.predict_proba(data_test))


class ScoringMethodNotFoundError(Exception):
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
        message="Input scoring method is not an allowed value.",
    ) -> None:
        self.input_value = input_value
        self.allowed = allowed
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f"{{self.message}} Allowed values: {self.allowed}. Got: {self.input_value}."


class DecoderNotFoundError(Exception):
    """Exception raised when invalid Decoder is passed.

    Attributes:
        input_value -- input which caused the error
        allowed -- allowed input types
        message -- explanation of the error
    """

    def __init__(
        self,
        input_value,
        allowed,
        message="Input decoding model is not an allowed value.",
    ) -> None:
        self.input_value = input_value
        self.allowed = allowed.values
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f"{{self.message}} Allowed values: {self.allowed}. Got: {self.input_value}."


@dataclass
class CATB(Decoder):
    """Class for CatBoostClassifier implementation."""

    def __post_init__(self):
        self.model = CatBoostClassifier(
            loss_function="MultiClass",
            verbose=False,
            use_best_model=True,
            eval_metric="MultiClass",
        )

    def fit(self, data: np.ndarray, labels: np.ndarray, groups) -> None:
        """"""
        self.data_train = data
        self.labels_train = labels
        self.groups_train = groups

        if self.optimize:
            self.model = self._bayesian_optimization()

        # Train outer model

        (
            self.data_train,
            self.labels_train,
            eval_set,
        ) = self._get_validation_split(
            self.data_train,
            self.labels_train,
            self.groups_train,
            train_size=0.75,
        )

        (
            self.data_train,
            self.labels_train,
            sample_weight,
        ) = self._balance_samples(
            self.data_train, self.labels_train, self.balancing
        )

        self.model.fit(
            self.data_train,
            self.labels_train,
            eval_set=eval_set,
            early_stopping_rounds=25,
            sample_weight=sample_weight,
            verbose=False,
        )

    def _bayesian_optimization(self):
        """Estimate optimal model parameters using bayesian optimization."""
        bo = BayesianOptimization(
            self._bo_tune,
            {
                "max_depth": (4, 10),
                "learning_rate": (0.003, 0.3),
                "bagging_temperature": (0.0, 1.0),
                "l2_leaf_reg": (1, 30),
                "random_strength": (0.01, 1.0),
            },
        )
        bo.maximize(init_points=10, n_iter=20, acq="ei")
        params = bo.max["params"]
        params["max_depth"] = round(params["max_depth"])
        return CatBoostClassifier(
            iterations=200,
            loss_function="MultiClass",
            verbose=False,
            use_best_model=True,
            eval_metric="MultiClass",
            max_depth=params["max_depth"],
            learning_rate=params["learning_rate"],
            random_strength=params["random_strength"],
            bagging_temperature=params["bagging_temperature"],
            l2_leaf_reg=params["l2_leaf_reg"],
        )

    def _bo_tune(
        self,
        max_depth,
        learning_rate,
        bagging_temperature,
        l2_leaf_reg,
        random_strength,
    ):
        # Cross validating with the specified parameters in 5 folds
        cv_inner = GroupShuffleSplit(
            n_splits=3, train_size=0.66, random_state=42
        )
        scores = []
        for train_index, test_index in cv_inner.split(
            self.data_train, self.labels_train, self.groups_train
        ):
            X_tr, X_te = (
                self.data_train[train_index],
                self.data_train[test_index],
            )
            y_tr, y_te = (
                self.labels_train[train_index],
                self.labels_train[test_index],
            )
            groups_split = self.groups_train[train_index]
            val_inner_split = GroupShuffleSplit(
                n_splits=1, train_size=0.75, random_state=41
            )
            for train_ind, val_ind in val_inner_split.split(
                X_tr, y_tr, groups_split
            ):
                X_tr, X_va = X_tr[train_ind], X_tr[val_ind]
                y_tr, y_va = y_tr[train_ind], y_tr[val_ind]
                eval_set_inner = [(X_va, y_va)]
                X_tr, y_tr, sample_weight = self._balance_samples(
                    X_tr, y_tr, self.balancing
                )
                inner_model = CatBoostClassifier(
                    iterations=100,
                    loss_function="MultiClass",
                    verbose=False,
                    eval_metric="MultiClass",
                    max_depth=round(max_depth),
                    learning_rate=learning_rate,
                    bagging_temperature=bagging_temperature,
                    l2_leaf_reg=l2_leaf_reg,
                    random_strength=random_strength,
                )
                inner_model.fit(
                    X_tr,
                    y_tr,
                    eval_set=eval_set_inner,
                    early_stopping_rounds=25,
                    sample_weight=sample_weight,
                    verbose=False,
                )
                y_probs = inner_model.predict_proba(X_te)
                score = log_loss(y_te, y_probs, labels=[0, 1])
                scores.append(score)
        # Return the negative MLOGLOSS
        return -1.0 * np.mean(scores)


@dataclass
class LDA(Decoder):
    """Class for applying Linear Discriminant Analysis using scikit-learn implementation."""

    def __post_init__(self):
        if self.balancing == "balance_weights":
            raise ValueError(
                "Sample weights cannot be balanced for Linear "
                "Discriminant Analysis. Please set `balance_weights` to"
                "either `oversample`, `undersample` or `None`."
            )
        if self.optimize:
            raise ValueError(
                "Hyperparameter optimization cannot be performed for this implementation of"
                "Linear Discriminant Analysis. Please set `optimize` to False."
            )

    def fit(self, data: np.ndarray, labels: np.ndarray, groups) -> None:
        """"""
        self.data_train, self.labels_train, _ = self._balance_samples(
            data, labels, self.balancing
        )
        self.model = LinearDiscriminantAnalysis(
            solver="lsqr", shrinkage="auto"
        )
        self.model.fit(self.data_train, self.labels_train)


@dataclass
class LR(Decoder):
    """Basic representation of class for finding and filtering files."""

    def fit(self, data: np.ndarray, labels: np.ndarray, groups) -> None:
        """"""
        self.data_train = data
        self.labels_train = labels
        self.groups_train = groups

        if self.optimize:
            self.model = self._bayesian_optimization()
        else:
            self.model = LogisticRegression(solver="newton-cg")

        self.data_train, self.labels_train, _ = self._balance_samples(
            data, labels, self.balancing
        )

        self.model.fit(self.data_train, self.labels_train)

    def _bayesian_optimization(self):
        """Estimate optimal model parameters using bayesian optimization."""
        bo = BayesianOptimization(self._bo_tune, {"C": (0.01, 1.0)})
        bo.maximize(init_points=10, n_iter=20, acq="ei")
        # Train outer model with optimized parameters
        params = bo.max["params"]
        # params['max_iter'] = int(params['max_iter'])
        return LogisticRegression(
            solver="newton-cg", max_iter=500, C=params["C"]
        )

    def _bo_tune(self, C):
        # Cross validating with the specified parameters in 5 folds
        cv_inner = GroupShuffleSplit(
            n_splits=3, train_size=0.66, random_state=42
        )
        scores = []

        for train_index, test_index in cv_inner.split(
            self.data_train, self.labels_train, self.groups_train
        ):
            X_tr, X_te = (
                self.data_train[train_index],
                self.data_train[test_index],
            )
            y_tr, y_te = (
                self.labels_train[train_index],
                self.labels_train[test_index],
            )
            X_tr, y_tr, sample_weight = self._balance_samples(
                X_tr, y_tr, self.balancing
            )
            inner_model = LogisticRegression(
                solver="newton-cg", C=C, max_iter=500
            )
            inner_model.fit(X_tr, y_tr, sample_weight=sample_weight)
            y_probs = inner_model.predict_proba(X_te)
            score = log_loss(y_te, y_probs, labels=[0, 1])
            scores.append(score)
        # Return the negative MLOGLOSS
        return -1.0 * np.mean(scores)


@dataclass
class Dummy(Decoder):
    """Dummy classifier implementation from scikit learn"""

    def fit(self, data: np.ndarray, labels: np.ndarray, groups) -> None:
        """"""
        self.data_train, self.labels_train, _ = self._balance_samples(
            data, labels, self.balancing
        )
        self.model = DummyClassifier(strategy="uniform")
        self.model.fit(self.data_train, self.labels_train)

    def get_score(self, data_test: np.ndarray, label_test: np.ndarray):
        """Calculate score."""
        scores = [
            self.scoring(self.model, data_test, label_test)
            for _ in range(0, 100)
        ]
        return np.mean(scores)


@dataclass
class QDA(Decoder):
    """Class for applying Linear Discriminant Analysis using scikit-learn implementation."""

    def __post_init__(self):
        if self.balancing == "balance_weights":
            raise ValueError(
                "Sample weights cannot be balanced for Quadratic "
                "Discriminant Analysis. Please set `balance_weights` to"
                "either `oversample`, `undersample` or `None`."
            )
        if self.optimize:
            raise ValueError(
                "Hyperparameter optimization cannot be performed for this implementation of"
                "Quadratic Discriminant Analysis. Please set `optimize` to False."
            )

    def fit(self, data: np.ndarray, labels: np.ndarray, groups) -> None:
        """"""
        self.data_train, self.labels_train, _ = self._balance_samples(
            data, labels, self.balancing
        )
        self.model = QuadraticDiscriminantAnalysis()
        self.model.fit(self.data_train, self.labels_train)


@dataclass
class XGB(Decoder):
    """Basic representation of class for finding and filtering files."""


@dataclass
class SVC_Lin(Decoder):
    """"""


@dataclass
class SVC_Poly(Decoder):
    """"""


@dataclass
class SVC_RBF(Decoder):
    """"""


@dataclass
class SVC_Sig(Decoder):
    """"""


def classify_svm_lin(X_train, y_train, group_train, optimize, balance):
    """"""

    def bo_tune(C, tol):
        # Cross validating with the specified parameters in 5 folds
        cv_inner = GroupShuffleSplit(
            n_splits=3, train_size=0.66, random_state=42
        )
        scores = []
        for train_index, test_index in cv_inner.split(
            X_train, y_train, group_train
        ):
            X_tr, X_te = X_train[train_index], X_train[test_index]
            y_tr, y_te = y_train[train_index], y_train[test_index]
            X_tr, y_tr, sample_weight = _balance_samples(X_tr, y_tr, balance)
            inner_model = SVC(
                kernel="linear",
                C=C,
                max_iter=500,
                tol=tol,
                gamma="scale",
                shrinking=True,
                class_weight=None,
                probability=True,
                verbose=False,
            )
            inner_model.fit(X_tr, y_tr, sample_weight=sample_weight)
            y_probs = inner_model.predict_proba(X_te)
            score = log_loss(y_te, y_probs, labels=[0, 1])
            scores.append(score)
        # Return the negative MLOGLOSS
        return -1.0 * np.mean(scores)

    if optimize:
        # Perform Bayesian Optimization
        bo = BayesianOptimization(
            bo_tune, {"C": (pow(10, -1), pow(10, 1)), "tol": (1e-4, 1e-2)}
        )
        bo.maximize(init_points=10, n_iter=20, acq="ei")
        # Train outer model with optimized parameters
        params = bo.max["params"]
        # params['max_iter'] = 500
        model = SVC(
            kernel="linear",
            C=params["C"],
            max_iter=500,
            tol=params["tol"],
            gamma="scale",
            shrinking=True,
            class_weight=None,
            verbose=False,
        )
    else:
        # Use default values
        model = SVC(
            kernel="linear",
            gamma="scale",
            shrinking=True,
            class_weight=None,
            verbose=False,
        )
    # Train outer model
    X_train, y_train, sample_weight = _balance_samples(
        X_train, y_train, balance
    )
    model.fit(X_train, y_train, sample_weight=sample_weight)
    return model


def classify_svm_rbf(X_train, y_train, group_train, optimize, balance):
    """"""

    def bo_tune(C, tol):
        # Cross validating with the specified parameters in 5 folds
        cv_inner = GroupShuffleSplit(
            n_splits=3, train_size=0.66, random_state=42
        )
        scores = []
        for train_index, test_index in cv_inner.split(
            X_train, y_train, group_train
        ):
            X_tr, X_te = X_train[train_index], X_train[test_index]
            y_tr, y_te = y_train[train_index], y_train[test_index]
            X_tr, y_tr, sample_weight = _balance_samples(X_tr, y_tr, balance)
            inner_model = SVC(
                kernel="rbf",
                C=C,
                max_iter=500,
                tol=tol,
                gamma="scale",
                shrinking=True,
                class_weight=None,
                probability=True,
                verbose=False,
            )
            inner_model.fit(X_tr, y_tr, sample_weight=sample_weight)
            y_probs = inner_model.predict_proba(X_te)
            score = log_loss(y_te, y_probs, labels=[0, 1])
            scores.append(score)
        # Return the negative MLOGLOSS
        return -1.0 * np.mean(scores)

    if optimize:
        # Perform Bayesian Optimization
        bo = BayesianOptimization(
            bo_tune, {"C": (pow(10, -1), pow(10, 1)), "tol": (1e-4, 1e-2)}
        )
        bo.maximize(init_points=10, n_iter=20, acq="ei")
        # Train outer model with optimized parameters
        params = bo.max["params"]
        model = SVC(
            kernel="rbf",
            C=params["C"],
            max_iter=500,
            tol=params["tol"],
            gamma="scale",
            shrinking=True,
            class_weight=None,
            verbose=False,
        )
    else:
        # Use default values
        model = SVC(
            kernel="rbf",
            gamma="scale",
            shrinking=True,
            class_weight=None,
            verbose=False,
        )
    # Train outer model
    X_train, y_train, sample_weight = _balance_samples(
        X_train, y_train, balance
    )
    model.fit(X_train, y_train, sample_weight=sample_weight)
    return model


balance = "undersample"
optimize = False


def classify_svm_poly(X_train, y_train, group_train):
    """"""

    def bo_tune(C, tol):
        # Cross validating with the specified parameters in 5 folds
        cv_inner = GroupShuffleSplit(
            n_splits=3, train_size=0.66, random_state=42
        )
        scores = []
        for train_index, test_index in cv_inner.split(
            X_train, y_train, group_train
        ):
            X_tr, X_te = X_train[train_index], X_train[test_index]
            y_tr, y_te = y_train[train_index], y_train[test_index]
            X_tr, y_tr, sample_weight = _balance_samples(X_tr, y_tr, balance)
            inner_model = SVC(
                kernel="poly",
                C=C,
                max_iter=500,
                tol=tol,
                gamma="scale",
                shrinking=True,
                class_weight=None,
                probability=True,
                verbose=False,
            )
            inner_model.fit(X_tr, y_tr, sample_weight=sample_weight)
            y_probs = inner_model.predict_proba(X_te)
            score = log_loss(y_te, y_probs, labels=[0, 1])
            scores.append(score)
        # Return the negative MLOGLOSS
        return -1.0 * np.mean(scores)

    if optimize:
        # Perform Bayesian Optimization
        bo = BayesianOptimization(
            bo_tune, {"C": (pow(10, -1), pow(10, 1)), "tol": (1e-4, 1e-2)}
        )
        bo.maximize(init_points=10, n_iter=20, acq="ei")
        # Train outer model with optimized parameters
        params = bo.max["params"]
        model = SVC(
            kernel="poly",
            C=params["C"],
            max_iter=500,
            tol=params["tol"],
            gamma="scale",
            shrinking=True,
            class_weight=None,
            verbose=False,
        )
    else:
        # Use default values
        model = SVC(
            kernel="poly",
            gamma="scale",
            shrinking=True,
            class_weight=None,
            verbose=False,
        )
    # Train outer model
    X_train, y_train, sample_weight = _balance_samples(
        X_train, y_train, balance
    )
    model.fit(X_train, y_train, sample_weight=sample_weight)
    return model


def classify_svm_sig(X_train, y_train, group_train, optimize, balance):
    """"""

    def bo_tune(C, tol):
        # Cross validating with the specified parameters in 5 folds
        cv_inner = GroupShuffleSplit(
            n_splits=3, train_size=0.66, random_state=42
        )
        scores = []
        for train_index, test_index in cv_inner.split(
            X_train, y_train, group_train
        ):
            X_tr, X_te = X_train[train_index], X_train[test_index]
            y_tr, y_te = y_train[train_index], y_train[test_index]
            X_tr, y_tr, sample_weight = _balance_samples(X_tr, y_tr, balance)
            inner_model = SVC(
                kernel="sigmoid",
                C=C,
                max_iter=500,
                tol=tol,
                gamma="auto",
                shrinking=True,
                class_weight=None,
                probability=True,
                verbose=False,
            )
            inner_model.fit(X_tr, y_tr, sample_weight=sample_weight)
            y_probs = inner_model.predict_proba(X_te)
            score = log_loss(y_te, y_probs, labels=[0, 1])
            scores.append(score)
        # Return the negative MLOGLOSS
        return -1.0 * np.mean(scores)

    if optimize:
        # Perform Bayesian Optimization
        bo = BayesianOptimization(
            bo_tune, {"C": (pow(10, -1), pow(10, 1)), "tol": (1e-4, 1e-2)}
        )
        bo.maximize(init_points=10, n_iter=20, acq="ei")
        # Train outer model with optimized parameters
        params = bo.max["params"]
        model = SVC(
            kernel="sigmoid",
            C=params["C"],
            max_iter=500,
            tol=params["tol"],
            gamma="auto",
            shrinking=True,
            class_weight=None,
            verbose=False,
        )
    else:
        # Use default values
        model = SVC(
            kernel="sigmoid",
            gamma="scale",
            shrinking=True,
            class_weight=None,
            verbose=False,
        )
    # Train outer model
    X_train, y_train, sample_weight = _balance_samples(
        X_train, y_train, balance
    )
    model.fit(X_train, y_train, sample_weight=sample_weight)
    return model


def classify_xgb(X_train, y_train, group_train, optimize, balance):
    """"""

    def bo_tune(max_depth, gamma, learning_rate, subsample, colsample_bytree):
        # Cross validating with the specified parameters in n_splits folds
        cv_inner = GroupShuffleSplit(
            n_splits=3, train_size=0.66, random_state=42
        )
        scores = []
        for train_index, test_index in cv_inner.split(
            X_train, y_train, group_train
        ):
            X_tr = np.ascontiguousarray(X_train[train_index])
            X_te = np.ascontiguousarray(X_train[test_index])
            y_tr, y_te = y_train[train_index], y_train[test_index]
            groups_split = group_train[train_index]
            val_inner_split = GroupShuffleSplit(
                n_splits=1, train_size=0.8, random_state=41
            )
            for train_ind, val_ind in val_inner_split.split(
                X_tr, y_tr, groups_split
            ):
                X_tr, X_va = X_tr[train_ind], X_tr[val_ind]
                y_tr, y_va = y_tr[train_ind], y_tr[val_ind]
                X_tr, y_tr, sample_weight = _balance_samples(
                    X_tr, y_tr, balance
                )
                eval_set_inner = [(X_va, y_va)]
                inner_model = XGBClassifier(
                    objective="binary:logistic",
                    use_label_encoder=False,
                    eval_metric="logloss",
                    n_estimators=200,
                    gamma=gamma,
                    learning_rate=learning_rate,
                    max_depth=int(max_depth),
                    colsample_bytree=colsample_bytree,
                    subsample=subsample,
                )
                inner_model.fit(
                    X_tr,
                    y_tr,
                    eval_set=eval_set_inner,
                    early_stopping_rounds=10,
                    sample_weight=sample_weight,
                    verbose=False,
                )
                y_probs = inner_model.predict_proba(X_te)
                score = log_loss(y_te, y_probs, labels=[0, 1])
                scores.append(score)
        # Return the negative MLOGLOSS
        return -1.0 * np.mean(scores)

    if optimize:
        # Perform Bayesian Optimization
        xgb_bo = BayesianOptimization(
            bo_tune,
            {
                "max_depth": (4, 10),
                "gamma": (0, 1),
                "learning_rate": (0.001, 0.3),
                "colsample_bytree": (0.1, 1),
                "subsample": (0.8, 1),
            },
        )
        xgb_bo.maximize(init_points=10, n_iter=20, acq="ei")
        # Train outer model with optimized parameters
        params = xgb_bo.max["params"]
        params["max_depth"] = int(params["max_depth"])
        model = XGBClassifier(
            objective="binary:logistic",
            use_label_encoder=False,
            n_estimators=200,
            eval_metric="logloss",
            gamma=params["gamma"],
            learning_rate=params["learning_rate"],
            max_depth=params["max_depth"],
            subsample=params["subsample"],
            colsample_bytree=params["colsample_bytree"],
        )
    else:
        # Use default values
        model = XGBClassifier(
            objective="binary:logistic",
            use_label_encoder=False,
            n_estimators=200,
            eval_metric="logloss",
        )
    # Train outer model
    val_split = GroupShuffleSplit(n_splits=1, train_size=0.8)
    for train_ind, val_ind in val_split.split(X_train, y_train, group_train):
        X_val = np.ascontiguousarray(X_train[val_ind])
        X_train = np.ascontiguousarray(X_train[train_ind])
        y_train, y_val = (
            np.ascontiguousarray(y_train[train_ind]),
            np.ascontiguousarray(y_train[val_ind]),
        )
        eval_set = [(X_val, y_val)]
        X_train, y_train, sample_weight = _balance_samples(
            X_train, y_train, balance
        )
        model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            early_stopping_rounds=10,
            sample_weight=sample_weight,
            verbose=False,
        )
    return model


def _balance_samples(test1, test2, test3):
    """"""
    return test1, test2, test3

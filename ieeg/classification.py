import math
import os

from matplotlib import pyplot as plt
from numba import njit

import numpy as np
import pandas as pd
from scipy.stats import zscore

from bayes_opt import BayesianOptimization
from catboost import CatBoostClassifier
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import mne_bids
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, log_loss
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    LeaveOneGroupOut,
)
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.svm import LinearSVC, SVC
from xgboost import XGBClassifier


def balance_samples(data, target, method="oversample"):
    """Balance class sizes to create equal class distributions.

    Parameters
    ----------
    data : array_like
        Feature array of shape (n_features, n_samples)
    target : array_like
        Array of class disribution of shape (n_samples, )
    method : {'oversample', 'undersample', 'weight'}
        Method to be used for rebalancing classes. 'oversample' will upsample
        the class with less samples. 'undersample' will downsample the class
        with more samples. 'weight' will generate balanced class weights.
        Default: 'oversample'

    Returns
    -------
    data : numpy.array
        Rebalanced feature array of shape (n_features, n_samples)
    target : numpy.array
        Corresponding class distributions. Class sizes are now evenly balanced.
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


def train_model(
    classifier,
    X_train,
    X_test,
    y_train,
    y_test,
    groups_train,
    optimize,
    balance,
):
    """"""
    if "catboost" in classifier:
        model = classify_catboost(
            X_train, X_test, y_train, y_test, groups_train, optimize, balance
        )
    elif "lda" in classifier:
        model = classify_lda(X_train, y_train, balance)
    elif "lin_svm" in classifier:
        model = classify_lin_svm(
            X_train, y_train, groups_train, optimize, balance
        )
    elif "lr" in classifier:
        model = classify_lr(X_train, y_train, groups_train, optimize, balance)
    elif "svm_lin" in classifier:
        model = classify_svm_lin(
            X_train, y_train, groups_train, optimize, balance
        )
    elif "svm_rbf" in classifier:
        model = classify_svm_rbf(
            X_train, y_train, groups_train, optimize, balance
        )
    elif "svm_poly" in classifier:
        model = classify_svm_poly(
            X_train, y_train, groups_train, optimize, balance
        )
    elif "svm_sig" in classifier:
        model = classify_svm_sig(
            X_train, y_train, groups_train, optimize, balance
        )
    elif "xgb" in classifier:
        model = classify_xgb(X_train, y_train, groups_train, optimize, balance)
    else:
        raise ValueError(f"Classifier not found: {classifier}")
    return model


def classify_catboost(
    X_train, X_test, y_train, y_test, group_train, optimize, balance
):
    """"""

    def bo_tune(
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
            X_train, y_train, group_train
        ):
            X_tr, X_te = X_train[train_index], X_train[test_index]
            y_tr, y_te = y_train[train_index], y_train[test_index]
            groups_split = group_train[train_index]
            val_inner_split = GroupShuffleSplit(
                n_splits=1, train_size=0.75, random_state=41
            )
            for train_ind, val_ind in val_inner_split.split(
                X_tr, y_tr, groups_split
            ):
                X_tr, X_va = X_tr[train_ind], X_tr[val_ind]
                y_tr, y_va = y_tr[train_ind], y_tr[val_ind]
                eval_set_inner = [(X_va, y_va)]
                X_tr, y_tr, sample_weight = balance_samples(
                    X_tr, y_tr, balance
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

    if optimize:
        # Perform Bayesian Optimization
        bo = BayesianOptimization(
            bo_tune,
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
        model = CatBoostClassifier(
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
    else:
        # Use default values
        model = CatBoostClassifier(
            loss_function="MultiClass",
            verbose=False,
            use_best_model=True,
            eval_metric="MultiClass",
        )
    # Train outer model
    sample_weight = None
    val_split = GroupShuffleSplit(n_splits=1, train_size=0.75)
    for train_ind, val_ind in val_split.split(X_train, y_train, group_train):
        X_train, X_val = X_train[train_ind], X_train[val_ind]
        y_train, y_val = y_train[train_ind], y_train[val_ind]
        eval_set = [(X_val, y_val)]
        X_train, y_train, sample_weight = balance_samples(
            X_train, y_train, balance
        )
        model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            early_stopping_rounds=25,
            sample_weight=sample_weight,
            verbose=False,
        )
    return model


def classify_lda(X_train, y_train, balance):
    """"""
    if balance == "weight":
        raise ValueError(
            "Sample weights cannot be balanced for Linear "
            "Discriminant Analysis. Please set `balance` to"
            "either `None`, `oversample` or `undersample`."
        )
    X_train, y_train, sample_weight = balance_samples(
        X_train, y_train, balance
    )
    model = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    model.fit(X_train, y_train)
    return model


def classify_lr(X_train, y_train, group_train, optimize, balance):
    """"""

    def bo_tune(C):
        # Cross validating with the specified parameters in 5 folds
        cv_inner = GroupShuffleSplit(
            n_splits=3, train_size=0.66, random_state=42
        )
        scores = []
        inner_model = LogisticRegression(solver="newton-cg", C=C, max_iter=500)
        for train_index, test_index in cv_inner.split(
            X_train, y_train, group_train
        ):
            X_tr, X_te = X_train[train_index], X_train[test_index]
            y_tr, y_te = y_train[train_index], y_train[test_index]
            X_tr, y_tr, sample_weight = balance_samples(X_tr, y_tr, balance)
            inner_model.fit(X_tr, y_tr, sample_weight=sample_weight)
            y_probs = inner_model.predict_proba(X_te)
            score = log_loss(y_te, y_probs, labels=[0, 1])
            scores.append(score)
        # Return the negative MLOGLOSS
        return -1.0 * np.mean(scores)

    if optimize:
        # Perform Bayesian Optimization
        bo = BayesianOptimization(bo_tune, {"C": (0.01, 1.0)})
        bo.maximize(init_points=10, n_iter=20, acq="ei")
        # Train outer model with optimized parameters
        params = bo.max["params"]
        # params['max_iter'] = int(params['max_iter'])
        model = LogisticRegression(
            solver="newton-cg", max_iter=500, C=params["C"]
        )
    else:
        # use default values
        model = LogisticRegression(solver="newton-cg")
    # Train outer model
    X_train, y_train, sample_weight = balance_samples(
        X_train, y_train, balance
    )
    model.fit(X_train, y_train, sample_weight=sample_weight)
    return model


def classify_lin_svm(X_train, y_train, group_train, optimize, balance):
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
            X_tr, y_tr, sample_weight = balance_samples(X_tr, y_tr, balance)
            inner_model = LinearSVC(
                penalty="l2",
                fit_intercept=True,
                C=C,
                max_iter=500,
                tol=tol,
                shrinking=True,
                class_weight=None,
                probability=True,
                verbose=False,
            )
            inner_model.fit(X_tr, y_tr, sample_weight=sample_weight)
            # cal = CalibratedClassifierCV(base_estimator=inner_model,
            #                            cv='prefit')
            # cal.fit(X_tr, y_tr)
            # y_probs = cal.predict_proba(X_te)
            # score = log_loss(y_te, y_probs, labels=[0, 1])
            # scores.append(score)
        # Return the negative MLOGLOSS
        return -1.0 * np.mean(scores)

    if optimize:
        # Perform Bayesian Optimization
        bo = BayesianOptimization(
            bo_tune, {"C": (1e-1, 1e1), "tol": (1e-5, 1e-3)}
        )
        bo.maximize(init_points=10, n_iter=20, acq="ei")
        # Train outer model with optimized parameters
        params = bo.max["params"]
        # params['max_iter'] = 500
        model = LinearSVC(
            penalty="l2",
            fit_intercept=True,
            C=params["C"],
            max_iter=500,
            tol=params["tol"],
            shrinking=True,
            class_weight=None,
            verbose=False,
        )
    else:
        # Use default values
        model = LinearSVC(
            penalty="l2", fit_intercept=True, class_weight=None, verbose=False
        )
    # Train outer model
    X_train, y_train, sample_weight = balance_samples(
        X_train, y_train, balance
    )
    model.fit(X_train, y_train, sample_weight=sample_weight)
    return model


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
            X_tr, y_tr, sample_weight = balance_samples(X_tr, y_tr, balance)
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
    X_train, y_train, sample_weight = balance_samples(
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
            X_tr, y_tr, sample_weight = balance_samples(X_tr, y_tr, balance)
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
    X_train, y_train, sample_weight = balance_samples(
        X_train, y_train, balance
    )
    model.fit(X_train, y_train, sample_weight=sample_weight)
    return model


def classify_svm_poly(X_train, y_train, group_train, optimize, balance):
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
            X_tr, y_tr, sample_weight = balance_samples(X_tr, y_tr, balance)
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
    X_train, y_train, sample_weight = balance_samples(
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
            X_tr, y_tr, sample_weight = balance_samples(X_tr, y_tr, balance)
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
    X_train, y_train, sample_weight = balance_samples(
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
                X_tr, y_tr, sample_weight = balance_samples(
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
        y_train, y_val = np.ascontiguousarray(
            y_train[train_ind]
        ), np.ascontiguousarray(y_train[val_ind])
        eval_set = [(X_val, y_val)]
        X_train, y_train, sample_weight = balance_samples(
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


def generate_outpath(
    root,
    feature_file,
    classifier,
    target_beg,
    target_en,
    use_channels_,
    optimize,
    use_times,
):
    """"""
    clf_str = "_" + classifier + "_"

    target_str = (
        "movement_"
        if target_en == "MovementEnd"
        else "mot_intention_" + str(target_beg) + "_" + str(target_en) + "_"
    )
    ch_str = use_channels_ + "_chs_"
    opt_str = "opt_" if optimize else "no_opt_"
    out_name = (
        feature_file
        + clf_str
        + target_str
        + ch_str
        + opt_str
        + str(use_times * 100)
        + "ms"
    )
    return os.path.join(root, feature_file, out_name)


def get_all_files(
    path,
    suffix,
    get_bids=False,
    prefix=None,
    bids_root=None,
    verbose=False,
    extension=None,
):
    """Return all files in all (sub-)directories of path with given suffixes and prefixes (case-insensitive).

    Args:
        path (string)
        suffix (iterable): e.g. ["vhdr", "edf"] or ".json"
        get_bids (boolean): True if BIDS_Path type should be returned instead of string. Default: False
        bids_root (string/path): Path of BIDS root folder. Only required if get_bids=True.
        prefix (iterable): e.g. ["SelfpacedRota", "ButtonPress] (optional)

    Returns:
        filepaths (list of strings or list of BIDS_Path)
    """

    if isinstance(suffix, str):
        suffix = [suffix]
    if isinstance(prefix, str):
        prefix = [prefix]

    filepaths = []
    for root, dirs, files in os.walk(path):
        for file in files:
            for suff in suffix:
                if file.endswith(suff.lower()):
                    if not prefix:
                        filepaths.append(os.path.join(root, file))
                    else:
                        for pref in prefix:
                            if pref.lower() in file.lower():
                                filepaths.append(os.path.join(root, file))

    bids_paths = filepaths
    if get_bids:
        if not bids_root:
            print(
                "Warning: No root folder given. Please pass bids_root parameter to create a complete BIDS_Path object."
            )
        bids_paths = []
        for filepath in filepaths:
            entities = mne_bids.get_entities_from_fname(filepath)
            try:
                bids_path = mne_bids.BIDSPath(
                    subject=entities["subject"],
                    session=entities["session"],
                    task=entities["task"],
                    run=entities["run"],
                    acquisition=entities["acquisition"],
                    suffix=entities["suffix"],
                    extension=extension,
                    root=bids_root,
                )
            except ValueError as err:
                print(
                    f"ValueError while creating BIDS_Path object for file {filepath}: {err}"
                )
            else:
                bids_paths.append(bids_path)

    if verbose:
        if not bids_paths:
            print("No corresponding files found.")
        else:
            print("Corresponding files found:")
            for idx, file in enumerate(bids_paths):
                print(idx, ":", os.path.basename(file))

    return bids_paths


def discard_trial(baseline, data_artifacts):
    """"""
    if any((baseline <= 0.0, np.count_nonzero(data_artifacts))):
        discard = True
    else:
        discard = False
    return discard


def get_trial_data(
    data,
    events,
    ind,
    target_begin,
    target_end,
    rest_beg_ind,
    rest_end_ind,
    artifacts,
):
    """"""
    data_art = None
    if target_end == "MovementEnd":
        data_rest = data[
            events[ind] + rest_beg_ind : events[ind] + rest_end_ind
        ]
        data_target = data[events[ind] + target_begin : events[ind + 1]]
        if artifacts is not None:
            data_art = artifacts[events[ind] + target_begin : events[ind + 1]]
    else:
        data_rest = data[
            events[ind] + rest_beg_ind : events[ind] + rest_end_ind
        ]
        data_target = data[
            events[ind] + target_begin : events[ind] + target_end
        ]
        if artifacts is not None:
            data_art = artifacts[
                events[ind] + target_begin : events[ind] + target_end
            ]
    return data_rest, data_target, data_art


def get_baseline_period(events, event_ind, dist_onset, dist_end, artifacts):
    """"""
    ind_onset = events[event_ind] - dist_onset
    if event_ind != 0:
        ind_end = events[event_ind - 1] + dist_end
    else:
        ind_end = 0
    if ind_onset <= 0:
        baseline = 0
    else:
        baseline = ind_onset - ind_end
        if artifacts is not None:
            data_art = artifacts[ind_end:ind_onset]
            bool_art = np.flatnonzero(data_art)
            ind_art = bool_art[-1] if bool_art.size != 0 else 0
            baseline = baseline - ind_art
    return baseline


def get_feat_array(
    data,
    events,
    sfreq,
    target_begin,
    target_end,
    dist_onset,
    dist_end,
    artifacts=None,
    verbose=False,
):
    """"""
    dist_onset = int(dist_onset * sfreq)
    dist_end = int(dist_end * sfreq)

    rest_beg, rest_end = -5.0, -2.0
    rest_end_ind = int(rest_end * sfreq)
    target_begin = int(target_begin * sfreq)
    if target_end != "MovementEnd":
        target_end = int(target_end * sfreq)

    X, y, events_used, group_list, events_discard = [], [], [], [], []

    for i, ind in enumerate(np.arange(0, len(events), 2)):
        baseline_period = get_baseline_period(
            events, ind, dist_onset, dist_end, artifacts
        )
        rest_beg_ind = int(
            max(rest_end_ind - baseline_period, rest_beg * sfreq)
        )
        data_rest, data_target, data_art = get_trial_data(
            data,
            events,
            ind,
            target_begin,
            target_end,
            rest_beg_ind,
            rest_end_ind,
            artifacts,
        )
        if not discard_trial(baseline_period, data_art):
            X.extend((data_rest, data_target))
            y.extend((np.zeros(len(data_rest)), np.ones(len(data_target))))
            events_used.append(ind)
            group_list.append(np.full((len(data_rest) + len(data_target)), i))
        else:
            events_discard.append(ind)
    if verbose:
        print("No. of trials used: ", len(events_used))
    return (
        np.concatenate(X, axis=0).squeeze(),
        np.concatenate(y),
        np.array(events_used),
        np.concatenate(group_list),
        np.array(events_discard),
    )


def get_feat_array_prediction(data, events, events_used, sfreq, begin, end):
    """"""
    begin = int(begin * sfreq)
    end = int(end * sfreq)
    epochs = []
    for ind in events_used:
        epoch = data[events[ind] + begin : events[ind] + end + 1]
        if len(epoch) == end - begin + 1:
            epochs.append(epoch.squeeze())
        else:
            print(
                f"Length mismatch of epochs. Got: {len(epoch)}, expected: "
                f"{end - begin + 1}. Event used: No. {ind + 1} of "
                f"{len(events)}."
            )
    if epochs:
        return np.stack(epochs, axis=0)
    else:
        return epochs


@njit
def clusterwise_pval_numba(p_arr, p_sig, n_perm):
    """Calculate significant clusters and their corresponding p-values.

    Based on:
    https://github.com/neuromodulation/wjn_toolbox/blob/4745557040ad26f3b8498ca5d0c5d5dece2d3ba1/mypcluster.m
    https://garstats.wordpress.com/2018/09/06/cluster/

    Arguments
    ---------
    p_arr :  array-like
        Array of p-values. WARNING: MUST be one-dimensional
    p_sig : float
        Significance level
    n_perm : int
        No. of random permutations for building cluster null-distribution

    Returns
    -------
    p : list of floats
        List of p-values for each cluster
    p_min_index : list of numpy array
        List of indices of each significant cluster
    """

    def cluster(iterable):
        """Cluster 1-D array of boolean values.

        Parameters
        ----------
        iterable : array-like of bool
            Array to be clustered.

        Returns
        -------
        cluster_labels : np.array
            Array of shape (len(iterable), 1), where each value indicates the
            number of the cluster. Values are 0 if the item does not belong to
            a cluster
        cluster_count : int
            Number of detected cluster. Corresponds to the highest value in
            cluster_labels
        """
        cluster_labels = np.zeros((len(iterable), 1))
        cluster_count = 0
        cluster_len = 0
        for idx, item in enumerate(iterable):
            if item:
                cluster_labels[idx] = cluster_count + 1
                cluster_len += 1
            elif cluster_len == 0:
                pass
            else:
                cluster_len = 0
                cluster_count += 1
        if cluster_len >= 1:
            cluster_count += 1
        return cluster_labels, cluster_count

    def calculate_null_distribution(p_arr_, p_sig_, n_perm_):
        """Calculate null distribution of clusters.

        Parameters
        ----------
        p_arr_ :  numpy array
            Array of p-values
        p_sig_ : float
            Significance level (p-value)
        n_perm_ : int
            No. of random permutations

        Returns
        -------
        r_per_arr : numpy array
            Null distribution of shape (n_perm_)
        """
        # loop through random permutation cycles
        r_per_arr = np.zeros(n_perm_)
        for r in range(n_perm_):
            r_per = np.random.randint(
                low=0, high=p_arr_.shape[0], size=p_arr_.shape[0]
            )
            labels_, n_clusters = cluster(p_arr_[r_per] <= p_sig_)

            cluster_ind = {}
            if n_clusters == 0:
                r_per_arr[r] = 0
            else:
                p_sum = np.zeros(n_clusters)
                for ind in range(n_clusters):
                    cluster_ind[ind] = np.where(labels_ == ind + 1)[0]
                    p_sum[ind] = np.sum(
                        np.asarray(1 - p_arr_[r_per])[cluster_ind[ind]]
                    )
                r_per_arr[r] = np.max(p_sum)
        return r_per_arr

    labels, num_clusters = cluster(p_arr <= p_sig)

    null_distr = calculate_null_distribution(p_arr, p_sig, n_perm)
    # Loop through clusters of p_val series or image
    clusters = []
    # Initialize empty list with specific data type for numba to work
    p_vals = [np.float64(x) for x in range(0)]
    # Cluster labels start at 1
    for cluster_i in range(num_clusters):
        index_cluster = np.where(labels == cluster_i + 1)[0]
        p_cluster_sum = np.sum(np.asarray(1 - p_arr)[index_cluster])
        p_val = (n_perm - np.sum(p_cluster_sum >= null_distr) + 1) / n_perm
        if p_val <= p_sig:
            clusters.append(index_cluster)
            p_vals.append(p_val)

    return p_vals, clusters


def events_from_label(label_data, verbose=False):
    """

    Parameters
    ----------
    label_data
    verbose

    Returns
    -------

    """
    label_diff = np.zeros_like(label_data, dtype=int)
    label_diff[1:] = np.diff(label_data)
    events_ = np.nonzero(label_diff)[0]
    if verbose:
        print(f"Number of events detected: {len(events_) / 2}")
    return events_


def get_picks(use_channels_, ch_names_, out_file_):
    """

    Parameters
    ----------
    use_channels_
    ch_names_
    out_file_

    Returns
    -------

    """
    if use_channels_ in ["single", "single_contralat"]:
        ch_picks_ = ch_names_
    elif use_channels_ == "all":
        ch_picks_ = ["ECOG", "LFP"]
    elif use_channels_ == "all_ipsilat":
        ch_picks_ = (
            ["ECOG", "LFP_L"] if "L_" in out_file_ else ["ECOG", "LFP_R"]
        )
    elif use_channels_ == "all_contralat":
        ch_picks_ = (
            ["ECOG", "LFP_R"] if "L_" in out_file_ else ["ECOG", "LFP_L"]
        )
    else:
        raise ValueError(
            f"use_channels keyword not identified: " f"{use_channels_}"
        )
    return ch_picks_


def get_target_df(targets, features_df):
    """

    Parameters
    ----------
    targets
    features_df

    Returns
    -------

    """
    i = 0
    target_df = pd.DataFrame()
    while len(target_df.columns) == 0:
        target_pick = targets[i]
        col_picks = [
            col for col in features_df.columns if target_pick in col.lower()
        ]
        for col in col_picks[:1]:
            target_df[col] = features_df[col]
        i += 1
    if len(col_picks[:1]) > 1:
        raise ValueError(f"Multiple targets found: {col_picks}")
    print("Target channel used: ", target_df.columns)
    return target_df


def inner_loop(
    ch_names,
    features,
    labels,
    groups,
    classifier,
    optimize,
    balance,
    cv=GroupShuffleSplit(n_splits=10, test_size=0.1),
):
    """"""
    results = []
    for train_ind, test_ind in cv.split(features.values, labels, groups):
        features_train, features_test = (
            features.iloc[train_ind],
            features.iloc[test_ind],
        )
        y_train, y_test = np.ascontiguousarray(
            labels[train_ind]
        ), np.ascontiguousarray(labels[test_ind])
        groups_train = groups[train_ind]
        for ch_name in ch_names:
            cols = [col for col in features_train.columns if ch_name in col]
            X_train = np.ascontiguousarray(features_train[cols].values)
            X_test = np.ascontiguousarray(features_test[cols].values)
            model = train_model(
                classifier,
                X_train,
                X_test,
                y_train,
                y_test,
                groups_train,
                optimize,
                balance,
            )
            y_pred = model.predict(X_test)
            accuracy = balanced_accuracy_score(y_test, y_pred)
            results.append([accuracy, ch_name])
    df = pd.DataFrame(data=results, columns=["accuracy", "channel"])
    results = []
    for ch in ch_names:
        df_chan = df[df["channel"] == ch]
        results.append([np.mean(df_chan["accuracy"].values), ch])
    df_new = pd.DataFrame(data=results, columns=["accuracy", "channel"])
    df_lfp = df_new[df_new["channel"].str.contains("LFP")]
    df_ecog = df_new[df_new["channel"].str.contains("ECOG")]
    best_ecog = df_ecog["channel"].loc[df_ecog["accuracy"].idxmax()]
    best_lfp = df_lfp["channel"].loc[df_lfp["accuracy"].idxmax()]
    print("Best channels:", *[best_ecog, best_lfp])
    return [best_ecog, best_lfp]


def run_prediction(
    features,
    target,
    events,
    artifacts,
    ch_names,
    target_begin,
    target_end,
    out_file,
    classifier="lda",
    dist_onset=2.0,
    dist_end=2.0,
    exceptions=None,
    excep_dist_end=0.0,
    optimize=False,
    balance="oversample",
    use_channels="single",
    verbose=False,
    pred_begin=-3.0,
    pred_end=3.0,
    pred_mode="classify",
    show_plot=False,
    save_plot=False,
):
    """Calculate classification performance and write to *.tsv file.

    Parameters
    ----------
    features : pd.DataFrame
        DataFrame of features to be used for classification, where each column
        is a different feature and each index a new sample.
    events : np.array
        Array of events of shape (n_trials * 2, ) where each even index
        [0, 2, 4, ...] marks the onset and each odd index [1, 3, 5, ...] marks
        the end of a trial (e.g. a movement).
    ch_names : list
        List of all channel names.
    target_begin : int | float
        Begin of target to be used for classification. Use e.g. 0.0 for target
        begin at movement onset or e.g. -1.5 for classifying movement intention
        starting -1.5s before motor onset.
    target_end : int | float | 'MovementEnd'
        End of target to be used for classification. Use 'MovementEnd' for
        target end at movement end or e.g. 0.0 for classifying movement
        intention up to motor onset.
    out_file : path | string
        Name and path of the file where classification performance is saved.
    classifier : {'lda', 'xgb', 'lr'}
        Method for classification. Use 'lda' for regularized shrinkage Linear
        Discriminant Analysis. Use 'xgb' for XGBoost classifier with
        hyperparameter optimization using Bayesian Optimization. Default is
        'lda'.
    dist_onset : int | float | default: 2.0
        Minimum distance before onset of current trial for label `rest`.
        Choosing a different value than 2.0 is currently not recommended for
        dist_onset.
    dist_end : int | float | default: 2.0
        Minimum distance after previous trial for label `rest`.

    Returns
    -------
    None
    """
    # Check for exception file
    if all((exceptions, any([exc in out_file for exc in exceptions]))):
        dist_end = excep_dist_end
        print("Exception file recognized: ", os.path.basename(out_file))

    # Check for plausability of events
    assert (len(events) / 2).is_integer(), (
        "Number of events is odd. Please " "check your data."
    )

    # Construct epoched array of features and labels using events
    data_epochs, labels, events_used, groups, events_discard = get_feat_array(
        features.values,
        events,
        sfreq=10,
        target_begin=target_begin,
        target_end=target_end,
        dist_onset=dist_onset,
        dist_end=dist_end,
        artifacts=artifacts,
        verbose=verbose,
    )

    # Initialize DataFrame from array
    feature_epochs = pd.DataFrame(data_epochs, columns=features.columns)

    # Initialize prediction results
    classifications = {"Movement": []}
    if use_channels == "single":
        classifications.update({ch: [] for ch in ch_names})
    elif use_channels == "single_contralat":
        side = "L_" if "R_" in out_file else "R_"
        ch_names = [ch for ch in ch_names if side in ch]
        classifications.update({ch: [] for ch in ch_names})
    else:
        classifications.update({ch: [] for ch in ["ECOG", "LFP"]})

    # Outer cross-validation
    cv_outer = LeaveOneGroupOut()
    # cv_outer = GroupKFold(5)
    results = []
    for fold, ind in enumerate(cv_outer.split(data_epochs, labels, groups)):
        if verbose:
            print(f"Fold no.: {fold + 1}")
        train_ind, test_ind = ind
        features_train, features_test = (
            feature_epochs.iloc[train_ind],
            feature_epochs.iloc[test_ind],
        )
        y_train = np.ascontiguousarray(labels[train_ind])
        y_test = np.ascontiguousarray(labels[test_ind])
        groups_train = groups[train_ind]

        # Get prediction epochs
        evs_test = np.unique(groups[test_ind]) * 2
        target_pred = get_feat_array_prediction(
            target.values,
            events,
            evs_test,
            sfreq=10,
            begin=pred_begin,
            end=pred_end,
        )
        if len(target_pred) == 0:
            pass
        else:
            if target_pred.ndim == 1:
                target_pred = np.expand_dims(target_pred, axis=0)
            for i, epoch in enumerate(target_pred):
                if abs(epoch.min()) > abs(epoch.max()):
                    target_pred[i] = epoch * -1.0
                target_pred[i] = (epoch - epoch.min()) / (
                    epoch.max() - epoch.min()
                )
            classifications["Movement"].extend(target_pred)

        # Handle which channels are used
        if use_channels == "single_best":
            ch_picks = sorted(
                inner_loop(
                    ch_names,
                    features_train,
                    y_train,
                    groups_train,
                    classifier,
                    optimize,
                    balance,
                    cv_outer,
                )
            )
        else:
            ch_picks = get_picks(use_channels, ch_names, out_file)

        # Infer channel types
        ch_types = ["ECOG" if "ECOG" in ch else "LFP" for ch in ch_picks]

        # Perform classification for each selected model
        for ch_name, ch_type in zip(ch_picks, ch_types):
            if verbose:
                print("Channel: ", ch_name)
            cols = [col for col in features_train.columns if ch_name in col]
            if verbose:
                print("Features used: ", len(cols))
            X_train = np.ascontiguousarray(features_train[cols].values)
            X_test = np.ascontiguousarray(features_test[cols].values)
            model = train_model(
                classifier,
                X_train,
                X_test,
                y_train,
                y_test,
                groups_train,
                optimize,
                balance,
            )
            imp = permutation_importance(
                model,
                X_test,
                y_test,
                scoring="balanced_accuracy",
                n_repeats=100,
                n_jobs=-1,
            )
            imp_scores = imp.importances_mean
            # imp_scores = model.coef_
            y_pred = model.predict(X_test)
            accuracy = balanced_accuracy_score(y_test, y_pred)
            y_prob = model.predict_proba(X_test)
            logloss = log_loss(y_test, y_prob)
            results.append(
                [
                    accuracy,
                    logloss,
                    fold,
                    ch_name,
                    imp_scores,
                    len(events_used),
                    len(events) // 2 - len(events_used),
                    events_discard,
                ]
            )

            # Perform predictions
            features_pred = get_feat_array_prediction(
                features[cols].values,
                events,
                evs_test,
                sfreq=10,
                begin=pred_begin,
                end=pred_end,
            )
            if len(features_pred) == 0:
                pass
            else:
                # Make predictions
                predictions = predict_epochs(model, features_pred, pred_mode)
                # Add prediction results to dictionary
                if use_channels in ["single", "single_contralat"]:
                    classifications[ch_name].extend(predictions)
                else:
                    classifications[ch_type].extend(predictions)

    # Save results, check if directory exists
    if not os.path.isdir(os.path.dirname(out_file)):
        os.makedirs(os.path.dirname(out_file))
    if verbose:
        print("Writing file: ", out_file, "\n")

    # Plot Predictions
    if show_plot or save_plot:
        # Perform classification for each selected model
        for ch_name in ch_picks:
            title = (
                ch_name
                + ": Class. target "
                + str(target_begin)
                + " - "
                + str(target_end)
            )
            plot_predictions(
                classifications[ch_name],
                label=classifications["Movement"],
                label_name="Movement",
                title=title,
                sfreq=10,
                axis_time=(pred_begin, pred_end),
                savefig=save_plot,
                show_plot=show_plot,
                filename=out_file,
            )

    # Save classification performance
    results_df = pd.DataFrame(
        data=results,
        columns=[
            "accuracy",
            "neg_logloss",
            "fold",
            "channel_name",
            "feature_importances",
            "trials_used",
            "trials_discarded",
            "IDs_discarded",
        ],
    )
    results_df.to_csv(out_file + "_results.tsv", sep="\t")

    # Save predictions
    classif_df = pd.DataFrame.from_dict(data=classifications)
    classif_df.to_csv(out_file + "_classif.tsv", sep="\t")


@njit
def permutation_numba_onesample(x, y, n_perm, two_tailed=True):
    """Perform permutation test with one-sample distribution.

    Parameters
    ----------
    x : array_like
        First distribution
    y : int or float
        Baseline against which to check for statistical significane
    n_perm : int
        Number of permutations
    two_tailed : bool, default: True
        Set to False if you would like to perform a one-sampled permutation
        test, else True
    two_tailed : bool, default: True
        Set to False if you would like to perform a one-tailed permutation
        test, else True

    Returns
    -------
    float
        Estimated difference of distribution from baseline
    float
        P-value of permutation test
    """
    if two_tailed is True:
        zeroed = x - y
        z = np.abs(np.mean(zeroed))
        p = np.empty(n_perm)
        # Run the simulation n_perm times
        for i in np.arange(n_perm):
            sign = np.random.choice(
                a=np.array([-1.0, 1.0]), size=len(x), replace=True
            )
            p[i] = np.abs(np.mean(zeroed * sign))
    else:
        zeroed = x - y
        z = np.mean(zeroed)
        p = np.empty(n_perm)
        # Run the simulation n_perm times
        for i in np.arange(n_perm):
            sign = np.random.choice(
                a=np.array([-1.0, 1.0]), size=len(x), replace=True
            )
            p[i] = np.mean(zeroed * sign)
        # Return p-value
    return z, (np.sum(p >= z) + 1) / (n_perm + 1)


@njit
def permutation_numba_twosample(x, y, n_perm, two_tailed=True):
    """Perform permutation test.

    Parameters
    ----------
    x : array_like
        First distribution
    y : array_like
        Second distribution
    n_perm : int
        Number of permutations
    two_tailed : bool, default: True
        Set to False if you would like to perform a one-sampled permutation
        test, else True
    two_tailed : bool, default: True
        Set to False if you would like to perform a one-tailed permutation
        test, else True

    Returns
    -------
    float
        Estimated difference of distribution means
    float
        P-value of permutation test
    """
    if two_tailed is True:
        z = np.abs(np.mean(x) - np.mean(y))
        pS = np.concatenate((x, y), axis=0)
        half = int(len(pS) / 2)
        p = np.empty(n_perm)
        # Run the simulation n_perm times
        for i in np.arange(0, n_perm):
            # Shuffle the data
            np.random.shuffle(pS)
            # Compute permuted absolute difference of the two sampled
            # distributions
            p[i] = np.abs(np.mean(pS[:half]) - np.mean(pS[half:]))
    else:
        z = np.mean(x) - np.mean(y)
        pS = np.concatenate((x, y), axis=0)
        half = int(len(pS) / 2)
        p = np.empty(n_perm)
        # Run the simulation n_perm times
        for i in np.arange(0, n_perm):
            # Shuffle the data
            np.random.shuffle(pS)
            # Compute permuted absolute difference of the two sampled
            # distributions
            p[i] = np.mean(pS[:half]) - np.mean(pS[half:])
    return z, (np.sum(p >= z) + 1) / (n_perm + 1)


def permutation(x, y, n_perm, two_tailed=True):
    """Perform permutation test.

    Parameters
    ----------
    x : array_like
        First distribution
    y : array_like or int or float
        Second distribution in the case of two-sampled test or baseline in the
        case of one-sampled test
    n_perm : int
        Number of permutations
    two_tailed : bool, default: True
        Set to False if you would like to perform a one-tailed permutation
        test, else True

    Returns
    -------
    float
        Estimated difference of distribution means or difference of single
        distribution mean from baseline
    float
        P-value of permutation test
    """
    if isinstance(y, (int, float)):
        # Perform one-sample permutation test
        if two_tailed:
            # Perform two-tailed permutation test
            zeroed = x - y
            z = np.abs(np.mean(zeroed))
            p = np.empty(n_perm)
            # Run the simulation n_perm times
            for i in np.arange(n_perm):
                sign = np.random.choice(
                    a=np.array([-1.0, 1.0]), size=len(x), replace=True
                )
                p[i] = np.abs(np.mean(zeroed * sign))
        else:
            # Perform one-tailed permutation test
            zeroed = x - y
            z = np.mean(zeroed)
            p = np.empty(n_perm)
            # Run the simulation n_perm times
            for i in np.arange(n_perm):
                sign = np.random.choice(
                    a=np.array([-1.0, 1.0]), size=len(x), replace=True
                )
                p[i] = np.mean(zeroed * sign)
    else:
        # Perform two-sample permutation test
        if two_tailed:
            # Perform two-tailed permutation test
            z = np.abs(np.mean(x) - np.mean(y))
            pS = np.concatenate((x, y), axis=0)
            half = int(len(pS) / 2)
            p = np.empty(n_perm)
            # Run the simulation n_perm times
            for i in np.arange(0, n_perm):
                # Shuffle the data
                np.random.shuffle(pS)
                # Compute permuted absolute difference of the two sampled
                # distributions
                p[i] = np.abs(np.mean(pS[:half]) - np.mean(pS[half:]))
        else:
            # Perform one-tailed permutation test
            z = np.mean(x) - np.mean(y)
            pS = np.concatenate((x, y), axis=0)
            half = int(len(pS) / 2)
            p = np.empty(n_perm)
            # Run the simulation n_perm times
            for i in np.arange(0, n_perm):
                # Shuffle the data
                np.random.shuffle(pS)
                # Compute permuted absolute difference of the two sampled
                # distributions
                p[i] = np.mean(pS[:half]) - np.mean(pS[half:])
        # Return p-value
    return z, (np.sum(p >= z) + 1) / (n_perm + 1)


@njit
def permutation_onesample_onetailed(x, y, n_perm):
    """"""
    # Perform one-tailed permutation test
    zeroed = x - y
    z = np.mean(zeroed)
    p = np.empty(n_perm)
    # Run the simulation n_perm times
    for i in np.arange(n_perm):
        sign = np.random.choice(
            a=np.array([-1.0, 1.0]), size=len(x), replace=True
        )
        p[i] = np.mean(zeroed * sign)
    return (np.sum(p >= z) + 1) / (n_perm + 1)


@njit
def permutation_onesample_twotailed(x, baseline, n_perm):
    """"""
    ## Initialize and pre-allocate
    zeroed = x - baseline
    sample_mean = np.abs(np.mean(zeroed))
    ## Run the simulation n_perm times
    z = np.empty(n_perm)
    for i in np.arange(n_perm):
        #  1. take n random draws from {-1, 1}, where len(x) is the length of
        #     the data to be tested
        mn = np.random.choice(
            a=np.array([-1.0, 1.0]), size=len(x), replace=True
        )
        #  2. assign the signs to the data and put them in a temporary variable
        flipped = zeroed * mn
        #  3. save the new data in an array
        z[i] = np.abs(np.mean(flipped))
    return (np.sum(z >= sample_mean) + 1) / (n_perm + 1)


@njit
def permutation_twosample_twotailed(x, y, n_perm):
    """Perform two-tailed permutation test of two distributions.

    Parameters
    ----------
    x : np array
        First distribution
    y : np array
        Second distribution
    n_perm : int
        Number of permutations

    Returns
    -------
    float
        estimated ground truth, here abs difference of distribution means
    float
        p-value of permutation test
    """
    # Compute ground truth difference
    sample_diff = np.abs(np.mean(x) - np.mean(y))
    half = len(x)
    # Initialize permutation
    pS = np.concatenate((x, y), axis=0)
    z = np.empty(n_perm)
    # Permutation loop
    for i in np.arange(0, n_perm):
        # Shuffle the data
        np.random.shuffle(pS)
        # Compute permuted absolute difference of the two sampled distributions
        z[i] = np.abs(np.mean(pS[:half]) - np.mean(pS[half:]))
    # Return p-value
    return (len(np.where(z >= sample_diff)[0]) + 1) / (n_perm + 1)


def plot_features(
    features,
    events,
    ch_names,
    path,
    sfreq,
    time_begin,
    time_end,
    dist_onset,
    dist_end,
):
    """"""
    dist_onset = int(dist_onset * sfreq)
    dist_end = int(dist_end * sfreq)
    samp_begin = int(time_begin * sfreq)
    samp_end = int(time_end * sfreq + 1)
    x = []
    data = features.values
    for i, ind in enumerate(np.arange(0, len(events), 2)):
        append = True
        if i == 0:
            data_plot = data[events[ind] + samp_begin : events[ind] + samp_end]
            if data_plot.shape[0] != samp_end - samp_begin:
                append = False
        elif (events[ind] - dist_onset) - (events[ind - 1] + dist_end) <= 0:
            append = False
        else:
            data_plot = data[events[ind] + samp_begin : events[ind] + samp_end]
            if data_plot.shape[0] != samp_end - samp_begin:
                append = False
        if append:
            x.extend(np.expand_dims(data_plot, axis=0))
    x = np.mean(np.stack(x, axis=0), axis=0)
    features = pd.DataFrame(x, columns=features.columns)

    n_rows = 2
    n_cols = int(math.ceil(len(ch_names) / n_rows))
    fig, axs = plt.subplots(
        figsize=(n_cols * 3, 5),
        nrows=n_rows,
        ncols=n_cols,
        dpi=300,
        sharex=False,
        sharey=True,
    )
    ind = 0
    ch_names.sort()
    for row in np.arange(n_rows):
        for col in np.arange(n_cols):
            if ind < len(ch_names):
                ch_name = ch_names[ind]
                cols = [col for col in features.columns if ch_name in col]
                # cols = [col for col in features.columns
                #       if ch_name in col and "diff" not in col]
                yticks = [
                    "Theta",
                    "Alpha",
                    "Low Beta",
                    "High Beta",
                    "Low Gamma",
                    "High Gamma",
                    "High Frequency Activity",
                    "Theta Derivation",
                    "Alpha Derivation",
                    "Low Beta Derivation",
                    "High Beta Derivation",
                    "Low Gamma Derivation",
                    "High Gamma Derivation",
                    "High Frequency Activity Derivation",
                ]
                # yticks = ["Theta", "Alpha", "Low Beta", "High Beta",
                #         "Low Gamma", "High Gamma", "High Frequency Activity"]
                x = features[cols].values
                ax = axs[row, col]
                ax.imshow(
                    zscore(x, axis=0).T,
                    cmap="viridis",
                    aspect="auto",
                    origin="lower",
                    vmin=-3.0,
                    vmax=3.0,
                )
                ax.set_yticks(np.arange(len(cols)))
                ax.set_yticklabels(yticks)
                ax.set_xticks(np.arange(0, x.shape[0] + 1, sfreq))
                ax.set_xticklabels(
                    np.arange(time_begin, time_end + 1, 1, dtype=int)
                )
                ax.set_title(str(ch_name))
                ind += 1
    for ax in axs.flat:
        ax.set(xlabel="Time [s]", ylabel="Features")
    fig.suptitle("Movement Aligned Features - Individual Channels")
    if not os.path.isdir(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_predictions(
    predictions,
    label,
    label_name,
    title,
    sfreq,
    axis_time,
    savefig=False,
    show_plot=False,
    filename=None,
):
    """"""
    predictions = np.stack(predictions, axis=0)
    label = np.stack(label, axis=0)
    fig, axs = plt.subplots(figsize=(5, 3))
    axs.plot(predictions.mean(axis=0), label="Predictions")
    axs.plot(label.mean(axis=0), color="m", label=label_name)
    axs.legend(loc="upper right")
    axs.set_xticks(np.arange(0, predictions.shape[1] + 1, sfreq))
    axs.set_xticklabels(np.arange(axis_time[0], axis_time[1] + 1, 1))
    axs.set_ylim(-0.02, 1.02)
    axs.set(xlabel="Time [s]", ylabel="Prediction Rate")
    fig.suptitle(
        title + "\n" + os.path.basename(filename).split("_ieeg")[0],
        fontsize="small",
    )
    fig.tight_layout()
    if savefig:
        fig.savefig(filename + ".png", dpi=300)
    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def predict_epochs(model, features, mode="classification"):
    """"""
    predictions = []
    if features.ndim < 3:
        np.expand_dims(features, axis=0)
    for trial in features:
        if mode == "classification":
            predictions.append(model.predict(trial))
        elif mode == "probability":
            predictions.append(model.predict_proba(trial)[:, 1])
        elif mode == "decision_function":
            predictions.append(model.decision_function(trial))
        else:
            raise ValueError(
                f"Only `classification`, `probability` or `decision_function` "
                f"are valid options for `mode`. Got {mode}."
            )
    return predictions


def get_feature_df(features, use_features, use_times):
    """

    Parameters
    ----------
    features
    use_features
    use_times

    Returns
    -------

    """
    # Extract features to use from dataframe
    column_picks = [
        col
        for col in features.columns
        if any([pick in col for pick in use_features])
    ]
    used_features = features[column_picks]

    # Initialize list of features to use
    feat_list = [
        used_features.rename(
            columns={col: col + "_100_ms" for col in used_features.columns}
        )
    ]

    # Use additional features from previous time points
    # use_times = 1 means no features from previous time points are
    # being used
    for s in np.arange(1, use_times):
        feat_list.append(
            used_features.shift(s, axis=0).rename(
                columns={
                    col: col + "_" + str((s + 1) * 100) + "_ms"
                    for col in used_features.columns
                }
            )
        )

    # Return final features dataframe
    return pd.concat(feat_list, axis=1).fillna(0.0)

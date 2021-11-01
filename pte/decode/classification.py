import math
import os

import numpy as np
import pandas as pd
from bayes_opt import BayesianOptimization
from catboost import CatBoostClassifier
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from matplotlib import pyplot as plt
from scipy.stats import zscore
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, log_loss
from sklearn.model_selection import GroupShuffleSplit
from sklearn.svm import SVC, LinearSVC
from sklearn.utils.class_weight import compute_sample_weight
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
    X_train, y_train, _ = balance_samples(X_train, y_train, balance)
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
        y_train, y_val = (
            np.ascontiguousarray(y_train[train_ind]),
            np.ascontiguousarray(y_train[val_ind]),
        )
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


def _generate_outpath(
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


def get_target_df(targets, features_df):
    """"""
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

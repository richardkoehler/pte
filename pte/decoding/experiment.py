"""Module for running a decoding experiment."""

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import BaseCrossValidator, GroupKFold

import pte
from pte.decoding.decode_abc import Decoder


@dataclass
class _Results:
    """Class for storing results of a single experiment."""

    target_name: str = field(repr=False)
    label_name: str = field(repr=False)
    ch_names: list[str] = field(repr=False)
    use_channels: str = field(repr=False)
    predictions: dict = field(init=False, default_factory=dict)
    scores: list = field(init=False, default_factory=list)
    features: dict = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        self._init_predictions()
        self._init_features()

    def _init_features(self) -> None:
        """Initialize features dictionary."""
        self.features = self._init_epoch_dict()
        self.features["ChannelNames"] = self.ch_names

    def _init_predictions(self) -> None:
        """Initialize predictions dictionary."""
        self.predictions = self._init_epoch_dict()

    def _init_epoch_dict(
        self,
    ) -> dict:
        """Initialize results dictionary."""
        results = {
            "Target": [],
            "TargetName": self.target_name,
            "Label": [],
            "LabelName": self.label_name,
        }
        if self.use_channels in [
            "all",
            "all_contralat",
            "all_ipsilat",
            "single_best",
            "single_best_contralat",
            "single_best_ipsilat",
        ]:
            results.update({ch: [] for ch in ["ECOG", "LFP"]})
        elif self.use_channels in [
            "single",
            "single_contralat",
            "single_ipsilat",
        ]:
            results.update({ch: [] for ch in self.ch_names})
        else:
            raise ValueError(
                f"Input value for `use_channels` not allowed. Got: "
                f"{self.use_channels}."
            )
        return results

    def update_labels(self, label_data: np.ndarray, label_name: str) -> None:
        """Update labels."""
        self.predictions = _add_label(
            data_dict=self.predictions,
            label_name=label_name,
            new_data=label_data,
        )
        self.features = _add_label(
            data_dict=self.features,
            label_name=label_name,
            new_data=label_data,
        )

    def update_scores(
        self,
        fold: int,
        ch_pick: str,
        score: Union[int, float],
        feature_importances: Sequence,
        feature_names: list[str],
        events_used: np.ndarray,
    ) -> None:
        """Update results."""
        if len(events_used) == 1:
            events_used = events_used[0]
        self.scores.append(
            [
                fold,
                ch_pick,
                score,
                feature_importances,
                feature_names,
                events_used,
            ]
        )

    def save(
        self,
        path: str,
        scoring: str,
        events: Union[np.ndarray, list],
        events_used: Union[np.ndarray, list],
    ) -> None:
        """Save results to given path."""
        # Save scores
        scores_df = pd.DataFrame(
            self.scores,
            columns=[
                "fold",
                "channel_name",
                scoring,
                "feature_importances",
                "features_used",
                "events_used",
            ],
            index=None,
        )
        scores_df.assign(
            **{
                "trials_used": len(events_used),
                "trials_discarded": len(events) // 2 - len(events_used),
            }
        )
        scores_df.to_csv(path + "_results.csv", sep=",", index=False)

        # Save predictions time-locked to trial onset
        with open(
            path + "_predictions_timelocked.json",
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(self.predictions, file)

        # Save features time-locked to trial onset
        with open(
            path + "_features_timelocked.json",
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(self.features, file)

    def update_predictions(
        self, predictions_data: list, ch_pick: str, ch_type: str
    ) -> None:
        """Update predictions and features."""
        self.predictions = _append_predictions(
            results=self.predictions,
            new_preds=predictions_data,
            use_channels=self.use_channels,
            ch_pick=ch_pick,
            ch_type=ch_type,
        )
        self.features = _append_predictions(
            results=self.features,
            new_preds=predictions_data,
            use_channels=self.use_channels,
            ch_pick=ch_pick,
            ch_type=ch_type,
        )


@dataclass
class Experiment:
    """Class for running prediction experiments."""

    features: pd.DataFrame
    target_df: pd.Series
    label: pd.Series
    ch_names: list[str]
    sfreq: int
    decoder: Optional[Decoder] = None
    side: Optional[str] = None
    artifacts: Optional[np.ndarray] = None
    bad_epochs: Optional[np.ndarray] = None
    scoring: str = "balanced_accuracy"
    feature_importance: Any = False
    target_begin: Union[str, float, int] = "trial_onset"
    target_end: Union[str, float, int] = "trial_end"
    dist_onset: Union[float, int] = 2.0
    dist_end: Union[float, int] = 2.0
    use_channels: str = "single"
    pred_mode: str = "classify"
    pred_begin: Union[float, int] = -3.0
    pred_end: Union[float, int] = 3.0
    cv_outer: BaseCrossValidator = GroupKFold(n_splits=5)
    cv_inner: BaseCrossValidator = GroupKFold(n_splits=5)
    verbose: bool = False
    feature_epochs: pd.DataFrame = field(init=False)
    data_epochs: np.ndarray = field(init=False)
    fold: int = field(init=False)
    labels: np.ndarray = field(init=False)
    groups: np.ndarray = field(init=False)
    events: np.ndarray = field(init=False)
    events_used: np.ndarray = field(init=False)
    events_discard: np.ndarray = field(init=False)
    results: _Results = field(init=False)

    def __post_init__(self) -> None:
        if self.target_begin == "trial_onset":
            self.target_begin = 0.0
        if self.target_end == "trial_onset":
            self.target_end = 0.0
        self.ch_names = _init_channel_names(
            self.ch_names, self.use_channels, self.side
        )

        self.results = _Results(
            target_name=self.target_df.name,
            label_name=self.label.name,
            ch_names=self.ch_names,
            use_channels=self.use_channels,
        )

        if self.decoder is None:
            self.decoder = pte.decoding.get_decoder(
                classifier="lda",
                scoring="balanced_accuracy",
                balancing="oversample",
            )

        # Calculate events from label
        self.events = _events_from_label(self.label.to_numpy(), self.verbose)

        # Check for plausability of events
        if not (len(self.events) / 2).is_integer():
            raise ValueError(
                f"Number of events is odd. Found {len(self.events) / 2} "
                f"events. Please check your data."
            )

        # Construct epoched array of features and labels using events
        (
            self.data_epochs,
            self.labels,
            self.events_used,
            self.events_discard,
            self.groups,
        ) = _get_feat_array(
            self.features.values,
            self.events,
            sfreq=self.sfreq,
            target_begin=self.target_begin,
            target_end=self.target_end,
            dist_onset=self.dist_onset,
            dist_end=self.dist_end,
            artifacts=self.artifacts,
            bad_epochs=self.bad_epochs,
        )

        # Initialize DataFrame from array
        self.feature_epochs = pd.DataFrame(
            self.data_epochs, columns=self.features.columns
        )

        self.fold = 0

    def run(self) -> None:
        """Calculate classification performance and out results."""
        # Outer cross-validation
        for train_ind, test_ind in self.cv_outer.split(
            self.data_epochs, self.labels, self.groups
        ):
            self._run_outer_cv(train_ind=train_ind, test_ind=test_ind)

    def save_results(self, path: Union[Path, str]) -> None:
        """Save results to given path."""
        path = Path(path)
        out_dir = path.parent
        path_str = str(path)
        # Save results, check if directory exists
        if not out_dir.is_dir():
            out_dir.mkdir(parents=True)
        if self.verbose:
            print("Writing results for file: ", path_str, "\n")

        self.results.save(
            path=path_str,
            scoring=self.scoring,
            events_used=self.events_used,
            events=self.events,
        )
        # Save all features used for classificaiton
        self.feature_epochs["Label"] = self.labels
        self.feature_epochs.to_csv(
            path_str + "_features_concatenated.csv",
        )

    def fit_and_save(self, path: Union[Path, str]) -> None:
        """Fit and save model using all good epochs."""
        # Handle which channels are used
        path = Path(path)
        out_dir = path.parent
        out_dir.mkdir(exist_ok=True, parents=True)

        ch_picks, ch_types = self._get_picks_and_types(
            features=self.feature_epochs,
            labels=self.labels,
            groups=self.groups,
            cross_val=self.cv_inner,
        )

        # Perform classification for each selected model
        for ch_pick, _ in zip(ch_picks, ch_types):
            cols = self._get_column_picks(self.feature_epochs, ch_pick)
            data_train = self.feature_epochs[cols]
            self.decoder.fit(
                data=data_train, labels=self.labels, groups=self.groups
            )
            basename = f"{path.name}_{ch_pick}.pickle"
            filename = str(path.with_name(basename))
            if self.verbose:
                print("Writing results for file: ", filename, "\n")
            with open(filename, "wb") as file:
                pickle.dump(self.decoder.model, file)

    def _update_results_labels(self, events_used: np.ndarray) -> None:
        """Update results with prediction labels."""
        for data, label_name in (
            (self.label.to_numpy(), "Label"),
            (self.target_df.to_numpy(), "Target"),
        ):
            epoch_data = _get_prediction_epochs(
                data=data,
                events=self.events,
                events_used=events_used,
                sfreq=self.sfreq,
                ind_begin=self.pred_begin,
                ind_end=self.pred_end,
                verbose=self.verbose,
            )
            if epoch_data is not None:
                self.results.update_labels(
                    label_data=epoch_data, label_name=label_name
                )

    def _run_outer_cv(self, train_ind: np.ndarray, test_ind: np.ndarray):
        """Run single outer cross-validation fold."""
        if self.verbose:
            print(f"Fold no.: {self.fold}")

        # Get training and testing data and labels
        features_train, features_test = (
            self.feature_epochs.iloc[train_ind],
            self.feature_epochs.iloc[test_ind],
        )
        y_train = self.labels[train_ind]
        y_test = self.labels[test_ind]
        groups_train = self.groups[train_ind]

        # Get prediction epochs
        evs_test = np.unique(self.groups[test_ind]) * 2

        self._update_results_labels(events_used=evs_test)

        # Handle which channels are used
        ch_picks, ch_types = self._get_picks_and_types(
            features=features_train,
            labels=y_train,
            groups=groups_train,
            cross_val=self.cv_inner,
        )

        # Perform classification for each selected model
        for ch_pick, ch_type in zip(ch_picks, ch_types):
            self._run_channel_pick(
                ch_pick,
                ch_type,
                features_train,
                features_test,
                y_train,
                y_test,
                groups_train,
                evs_test,
            )
        self.fold += 1

    def _run_channel_pick(
        self,
        ch_pick: str,
        ch_type: str,
        features_train: pd.DataFrame,
        features_test: pd.DataFrame,
        labels_train: np.ndarray,
        labels_test: np.ndarray,
        groups_train: np.ndarray,
        evs_test: np.ndarray,
    ) -> None:
        """Train model and save results for given channel picks"""
        cols = self._get_column_picks(features_train, ch_pick)
        data_train, data_test = features_train[cols], features_test[cols]

        self.decoder.fit(data_train, labels_train, groups_train)

        score = self.decoder.get_score(data_test, labels_test)

        feature_importances = _get_importances(
            feature_importance=self.feature_importance,
            decoder=self.decoder,
            data=data_test,
            label=labels_test,
            scoring=self.scoring,
        )

        self._update_results(
            ch_pick=ch_pick,
            ch_type=ch_type,
            events_used=evs_test,
            score=score,
            feature_importances=feature_importances,
            cols=cols,
        )

    def _get_column_picks(
        self,
        features: pd.DataFrame,
        channel_pick: str,
    ) -> list:
        """Return column picks given channel picksf rom features DataFrame."""
        col_picks = []
        for column in features.columns:
            if channel_pick in column:
                if any(ch_name in column for ch_name in self.ch_names):
                    col_picks.append(column)
        if self.verbose:
            print("No. of features used:", len(col_picks))
        return col_picks

    def _update_results(
        self,
        ch_pick: str,
        ch_type: str,
        events_used: np.ndarray,
        score: Union[float, int],
        feature_importances: list[Union[float, int]],
        cols: list[str],
    ) -> None:
        """Update results."""
        self.results.update_scores(
            fold=self.fold,
            ch_pick=ch_pick,
            score=score,
            feature_importances=feature_importances,
            feature_names=cols,
            events_used=events_used,
        )

        features_pred = _get_prediction_epochs(
            data=self.features[cols].values,
            events=self.events,
            events_used=events_used,
            sfreq=self.sfreq,
            ind_begin=self.pred_begin,
            ind_end=self.pred_end,
        )

        if features_pred is not None:
            new_preds = _predict_epochs(
                self.decoder.model, features_pred, self.pred_mode, columns=cols
            )
            self.results.update_predictions(
                predictions_data=new_preds, ch_pick=ch_pick, ch_type=ch_type
            )

    def _get_picks_and_types(
        self,
        features: pd.DataFrame,
        labels: np.ndarray,
        groups: np.ndarray,
        cross_val: BaseCrossValidator,
    ) -> tuple[list, list]:
        """Return channel picks and types."""
        if "single_best" in self.use_channels:
            ch_names = self._inner_loop(
                self.ch_names, features, labels, groups, cross_val
            )
        elif "all" in self.use_channels:
            ch_names = ["ECOG", "LFP"]
        else:
            ch_names = self.ch_names
        ch_types = ["ECOG" if "ECOG" in ch else "LFP" for ch in self.ch_names]
        return ch_names, ch_types

    def _inner_loop(
        self,
        ch_names: list[str],
        features: pd.DataFrame,
        labels: np.ndarray,
        groups: np.ndarray,
        cross_validator: BaseCrossValidator,
    ) -> list[str]:
        """Run inner cross-validation and return best ECOG and LFP channel."""
        results = {ch_name: [] for ch_name in ch_names}
        for train_ind, test_ind in cross_validator.split(
            features.values, labels, groups
        ):
            features_train, features_test = (
                features.iloc[train_ind],
                features.iloc[test_ind],
            )
            y_train, y_test = labels[train_ind], labels[test_ind]
            groups_train = groups[train_ind]
            for ch_name in ch_names:
                cols = [
                    col for col in features_train.columns if ch_name in col
                ]
                data_train = features_train[cols].to_numpy()
                data_test = features_test[cols].to_numpy()
                self.decoder.fit(data_train, y_train, groups_train)
                y_pred = self.decoder.model.predict(data_test)
                accuracy = balanced_accuracy_score(y_test, y_pred)
                results[ch_name].append(accuracy)
        lfp_results = {
            ch_name: np.mean(scores)
            for ch_name, scores in results.items()
            if "LFP" in ch_name
        }
        ecog_results = {
            ch_name: np.mean(scores)
            for ch_name, scores in results.items()
            if "ECOG" in ch_name
        }
        best_lfp = sorted(
            lfp_results.items(), key=lambda x: x[1], reverse=True
        )[0]
        best_ecog = sorted(
            ecog_results.items(), key=lambda x: x[1], reverse=True
        )[0]
        return [best_ecog, best_lfp]


def _get_importances(
    feature_importance: Union[int, bool],
    decoder: Decoder,
    data: pd.DataFrame,
    label: np.ndarray,
    scoring: str,
) -> list:
    """Calculate feature importances."""
    if not feature_importance:
        return []
    if feature_importance is True:
        return decoder.model.coef_
    if isinstance(feature_importance, int):
        imp_scores = permutation_importance(
            decoder.model,
            data,
            label,
            scoring=scoring,
            n_repeats=feature_importance,
            n_jobs=-1,
        ).importances_mean
        return imp_scores
    raise ValueError(
        f"`feature_importances` must be an integer or `False`. Got: "
        f"{feature_importance}."
    )


def _get_prediction_epochs(
    data: np.ndarray,
    events: Union[list, np.ndarray],
    events_used: np.ndarray,
    sfreq: Union[int, float],
    ind_begin: Union[int, float],
    ind_end: Union[int, float],
    verbose: bool = False,
) -> Optional[np.ndarray]:
    """Get epochs of data for making predictions."""
    ind_begin = int(ind_begin * sfreq)
    ind_end = int(ind_end * sfreq)
    epochs = []
    for ind in events_used:
        epoch = data[events[ind] + ind_begin : events[ind] + ind_end + 1]
        if len(epoch) == ind_end - ind_begin + 1:
            epochs.append(epoch.squeeze())
        else:
            if verbose:
                print(
                    f"Mismatch of epoch samples. Got: {len(epoch)} "
                    f"samples. Expected: {ind_end - ind_begin + 1} samples. "
                    f"Discarding epoch: No. {ind + 1} of {len(events)}."
                )
            else:
                pass
    if epochs:
        return np.stack(epochs, axis=0)
    return None


def _transform_side(side: str) -> str:
    """Transform given keyword (eg 'right') to search string (eg 'R_')."""
    if side == "right":
        return "R_"
    if side == "left":
        return "L_"
    raise ValueError(
        f"Invalid argument for `side`. Must be right " f"or left. Got: {side}."
    )


def _get_trial_data(
    data: np.ndarray,
    events: np.ndarray,
    event_ind: int,
    target_begin: int,
    target_end: Union[int, str],
    rest_beg_ind: int,
    rest_end_ind: int,
    artifacts: Optional[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get data of single trial for given event index."""
    data_art = None
    if target_end == "trial_end":
        data_rest = data[
            events[event_ind] + rest_beg_ind : events[event_ind] + rest_end_ind
        ]
        data_target = data[
            events[event_ind] + target_begin : events[event_ind + 1]
        ]
        if artifacts is not None:
            data_art = artifacts[
                events[event_ind] + target_begin : events[event_ind + 1]
            ]
    else:
        data_rest = data[
            events[event_ind] + rest_beg_ind : events[event_ind] + rest_end_ind
        ]
        data_target = data[
            events[event_ind] + target_begin : events[event_ind] + target_end
        ]
        if artifacts is not None:
            data_art = artifacts[
                events[event_ind]
                + target_begin : events[event_ind]
                + target_end
            ]
    if data_art is None:
        data_art = np.atleast_1d([])
    return data_rest, data_target, data_art


def _discard_trial(
    baseline: Union[int, float],
    data_artifacts: Optional[np.ndarray],
    index_epoch: int,
    bad_epochs: Optional[np.ndarray] = None,
) -> bool:
    """Decide if trial should be discarded."""
    if bad_epochs is None:
        bad_epochs = np.atleast_1d([])
    if any(
        (
            baseline <= 0.0,
            np.count_nonzero(data_artifacts),
            index_epoch in bad_epochs,
        )
    ):
        return True
    return False


def _predict_epochs(
    model: Any, features: np.ndarray, mode: str, columns: Optional[list] = None
) -> list[list]:
    """Make predictions for given feature epochs."""
    predictions = []
    if features.ndim < 3:
        np.expand_dims(features, axis=0)
    for trial in features:
        trial = pd.DataFrame(trial, columns=columns)
        if mode == "classification":
            pred = model.predict(trial).tolist()
        elif mode == "probability":
            pred = model.predict_proba(trial)[:, 1].tolist()
        elif mode == "decision_function":
            pred = model.decision_function(trial).tolist()
        else:
            raise ValueError(
                f"Only `classification`, `probability` or "
                f"`decision_function` are valid options for "
                f"`mode`. Got {mode}."
            )
        predictions.append(pred)
    return predictions


def _events_from_label(
    label_data: np.ndarray, verbose: bool = False
) -> np.ndarray:
    """Create array of events from given label data."""
    label_diff = np.zeros_like(label_data, dtype=int)
    label_diff[1:] = np.diff(label_data)
    if label_data[0] != 0:
        label_diff[0] = 1
    if label_data[-1] != 0:
        label_diff[-1] = -1
    events = np.nonzero(label_diff)[0]
    if verbose:
        print(f"Number of events detected: {len(events) / 2}")
    return events


def _append_predictions(
    results: dict,
    new_preds: list[list],
    use_channels: str,
    ch_pick: str,
    ch_type: str,
) -> dict:
    """Append new results to existing results."""
    if isinstance(new_preds, np.ndarray):
        new_preds = new_preds.tolist()
    # Add prediction results to dictionary
    if use_channels in ["single", "single_contralat", "single_ipsilat"]:
        results[ch_pick].extend(new_preds)
    else:
        results[ch_type].extend(new_preds)
    return results


def _add_label(data_dict: dict, label_name: str, new_data: np.ndarray) -> dict:
    """Append array of labels to classifications results."""
    if new_data.ndim == 0:
        new_data = np.expand_dims(new_data, axis=-1)
    for i, epoch in enumerate(new_data):
        # Invert array if necessary
        if abs(epoch.min()) > abs(epoch.max()):
            new_data[i] = epoch * -2.0
        # Perform min-max scaling
        new_data[i] = (epoch - epoch.min()) / (epoch.max() - epoch.min())
    data_dict[label_name].extend(new_data.tolist())
    return data_dict


def _init_channel_names(
    ch_names: list, use_channels: str, side: Optional[str] = None
) -> list:
    """Initialize channels to be used."""
    case_all = ["single", "single_best", "all"]
    case_contralateral = [
        "single_contralat",
        "single_best_contralat",
        "all_contralat",
    ]
    case_ipsilateral = [
        "single_ipsilat",
        "single_best_ipsilat",
        "all_ipsilat",
    ]
    if use_channels in case_all:
        return ch_names
    # If side is none but ipsi- or contralateral channels are selected
    if side is None:
        raise ValueError(
            f"`use_channels`: {use_channels} defines a hemisphere, but "
            f"`side` is not specified. Please pass `right` or `left` "
            f"side or set use_channels to any of: {(*case_all,)}."
        )
    side = _transform_side(side)
    if use_channels in case_contralateral:
        return [ch for ch in ch_names if side not in ch]
    if use_channels in case_ipsilateral:
        return [ch for ch in ch_names if side in ch]
    raise ValueError(
        f"Invalid argument for `use_channels`. Must be one of "
        f"{case_all+case_contralateral+case_ipsilateral}. Got: "
        f"{use_channels}."
    )


def _get_baseline_period(
    events: np.ndarray,
    event_ind: int,
    dist_onset: int,
    dist_end: int,
    artifacts: Optional[np.ndarray],
) -> int:
    """Return index where baseline period starts."""
    ind_onset: int = events[event_ind] - dist_onset
    if event_ind != 0:
        ind_end: int = events[event_ind - 1] + dist_end
    else:
        ind_end: int = 0
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


def _get_feat_array(
    data: np.ndarray,
    events: np.ndarray,
    sfreq: Union[int, float],
    target_begin: Union[float, int],
    target_end: Union[str, float, int] = "trial_end",
    dist_onset: Union[float, int] = 2.0,
    dist_end: Union[float, int] = 2.0,
    artifacts: Optional[np.ndarray] = None,
    bad_epochs: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Get features, labels, used and discarded events and groups by events."""
    dist_onset = int(dist_onset * sfreq)
    dist_end = int(dist_end * sfreq)

    rest_beg, rest_end = -5.0, -2.0
    rest_end_ind = int(rest_end * sfreq)
    target_begin = int(target_begin * sfreq)
    if isinstance(target_end, str):
        if target_end != "trial_end":
            raise ValueError(
                "`target_end` must be either a float, an"
                f"integer or 'trial_end'. Got: {target_end}."
            )
    else:
        target_end = int(target_end * sfreq)

    features, labels, events_used, events_discard, groups = (
        [],
        [],
        [],
        [],
        [],
    )

    for i, ind in enumerate(np.arange(0, len(events), 2)):
        baseline_period = _get_baseline_period(
            events, ind, dist_onset, dist_end, artifacts
        )
        rest_beg_ind = int(
            max(rest_end_ind - baseline_period, rest_beg * sfreq)
        )
        data_rest, data_target, data_art = _get_trial_data(
            data,
            events,
            ind,
            target_begin,
            target_end,
            rest_beg_ind,
            rest_end_ind,
            artifacts,
        )
        if not _discard_trial(
            baseline=baseline_period,
            data_artifacts=data_art,
            index_epoch=i,
            bad_epochs=bad_epochs,
        ):
            features.extend((data_rest, data_target))
            labels.extend(
                (np.zeros(len(data_rest)), np.ones(len(data_target)))
            )
            events_used.append(ind)
            groups.append(np.full((len(data_rest) + len(data_target)), i))
        else:
            events_discard.append(ind)
    return (
        np.concatenate(features, axis=0).squeeze(),
        np.concatenate(labels),
        np.array(events_used),
        np.array(events_discard),
        np.concatenate(groups),
    )

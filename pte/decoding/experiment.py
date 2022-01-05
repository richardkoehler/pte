"""Module for running a decoding experiment."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Union

import numpy as np
import pandas as pd
import sklearn
from sklearn.inspection import permutation_importance
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from pte.decoding.decode_abc import Decoder


@dataclass
class Results:
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

        self.features = self._init_epoch_dict(
            self.ch_names, self.use_channels, self.target_name
        )
        self.features["Label"] = []
        self.features["LabelName"] = self.label_name
        self.features["ChannelNames"] = self.ch_names

    def _init_predictions(self) -> None:
        """Initialize predictions dictionary."""

        self.predictions = self._init_epoch_dict(
            self.ch_names, self.use_channels, self.target_name
        )
        self.predictions["Label"] = []
        self.predictions["LabelName"] = self.label_name

    @staticmethod
    def _init_epoch_dict(
        ch_names: list, use_channels: str, target_name: str
    ) -> dict:
        """Initialize results dictionary."""
        results = {"Target": [], "TargetName": target_name}
        if use_channels in [
            "all",
            "all_contralat",
            "all_ipsilat",
            "single_best",
            "single_best_contralat",
            "single_best_ipsilat",
        ]:
            results.update({ch: [] for ch in ["ECOG", "LFP"]})
        elif use_channels in ["single", "single_contralat", "single_ipsilat"]:
            results.update({ch: [] for ch in ch_names})
        else:
            raise ValueError(
                f"Input value for `use_channels` not allowed. Got: "
                f"{use_channels}."
            )
        return results

    def update_labels(self, label_data: np.ndarray, label_name: str) -> None:
        """Update labels."""
        self.predictions = self._add_label(
            data_dict=self.predictions,
            label_name=label_name,
            new_data=label_data,
        )
        self.features = self._add_label(
            data_dict=self.features,
            label_name=label_name,
            new_data=label_data,
        )

    def update_scores(
        self,
        fold: int,
        ch_pick: str,
        score: Union[int, float],
        feature_importances: Iterable,
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
        events_discarded: Iterable,
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
            path + "_predictions_timelocked.json", "w", encoding="utf-8",
        ) as file:
            json.dump(self.predictions, file)

        # Save features time-locked to trial onset
        with open(
            path + "_features_timelocked.json", "w", encoding="utf-8",
        ) as file:
            json.dump(self.features, file)

    def update_predictions(
        self, predictions_data: list, ch_pick: str, ch_type: str
    ) -> None:
        """Update predictions."""
        self.predictions = self._append_predictions(
            results=self.predictions,
            new_preds=predictions_data,
            use_channels=self.use_channels,
            ch_pick=ch_pick,
            ch_type=ch_type,
        )
        self.features = self._append_predictions(
            results=self.features,
            new_preds=predictions_data,
            use_channels=self.use_channels,
            ch_pick=ch_pick,
            ch_type=ch_type,
        )

    @staticmethod
    def _append_predictions(
        results: dict,
        new_preds: list[list],
        use_channels: str,
        ch_pick: str,
        ch_type: str,
    ) -> dict:
        """Append new results to existing results."""
        if new_preds is None:
            return results
        if isinstance(new_preds, np.ndarray):
            new_preds = new_preds.tolist()
        # Add prediction results to dictionary
        if use_channels in ["single", "single_contralat", "single_ipsilat"]:
            results[ch_pick].extend(new_preds)
        else:
            results[ch_type].extend(new_preds)
        return results

    @staticmethod
    def _add_label(
        data_dict: dict, label_name: str, new_data: np.ndarray
    ) -> dict:
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


@dataclass
class Experiment:
    """Class for running prediction experiments."""

    features: pd.DataFrame
    target_df: pd.DataFrame
    label: pd.DataFrame
    ch_names: list[str]
    decoder: Optional[Decoder] = None
    side: Optional[str] = None
    artifacts: Optional[np.ndarray] = None
    bad_epochs: Optional[Iterable[int]] = None
    sfreq: int = None
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
    cv_outer: sklearn.model_selection.BaseCrossValidator = GroupKFold(
        n_splits=5
    )
    cv_inner: sklearn.model_selection.BaseCrossValidator = GroupKFold(
        n_splits=5
    )
    verbose: bool = False
    feature_epochs: pd.DataFrame = field(init=False)
    data_epochs: np.ndarray = field(init=False)
    fold: int = field(init=False)
    labels: np.ndarray = field(init=False)
    groups: np.ndarray = field(init=False)
    events: np.ndarray = field(init=False)
    events_used: np.ndarray = field(init=False)
    events_discard: np.ndarray = field(init=False)
    results: Results = field(init=False)

    def __post_init__(self) -> None:
        if self.target_begin == "trial_onset":
            self.target_begin = 0.0
        if self.target_end == "trial_onset":
            self.target_end = 0.0
        self.ch_names = self._init_channel_names(
            self.ch_names, self.use_channels, self.side
        )

        self.results = Results(
            target_name=self.target_df.columns[-1],
            label_name=self.label.name,
            ch_names=self.ch_names,
            use_channels=self.use_channels,
        )

        if self.bad_epochs is None:
            self.bad_epochs = np.atleast_1d([])

        # Calculate events from label
        self.events = self._events_from_label(self.label.values, self.verbose)

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
            self.groups,
            self.events_discard,
        ) = self._get_feat_array(
            self.features.values,
            self.events,
            sfreq=self.sfreq,
            target_begin=self.target_begin,
            target_end=self.target_end,
            dist_onset=self.dist_onset,
            dist_end=self.dist_end,
            artifacts=self.artifacts,
            bad_epochs=self.bad_epochs,
            verbose=self.verbose,
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

    def save(self, path: Union[Path, str]) -> None:
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
            events_discarded=self.events_discard,
        )
        # Save all features used for classificaiton
        self.feature_epochs["Label"] = self.labels
        self.feature_epochs.to_csv(path_str + "_features_concatenated.csv",)

    def _update_results_labels(self, events_used: np.ndarray) -> None:
        """Update results with prediction labels"""
        for data, label_name in (
            (self.label.values, "Label"),
            (self.target_df.values, "Target"),
        ):
            epoch_data = self._get_feat_array_prediction(
                data=data,
                events=self.events,
                events_used=events_used,
                sfreq=self.sfreq,
                begin=self.pred_begin,
                end=self.pred_end,
                verbose=self.verbose,
            )
            if epoch_data.size > 0:
                self.results.update_labels(
                    label_data=epoch_data, label_name=label_name
                )

    def _run_outer_cv(self, train_ind: np.ndarray, test_ind: np.ndarray):
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
        ch_picks = self._get_ch_picks(
            features=features_train,
            y=y_train,
            groups=groups_train,
            cv=self.cv_inner,
        )

        # Infer channel types
        ch_types = ["ECOG" if "ECOG" in ch else "LFP" for ch in ch_picks]

        # Perform classification for each selected model
        for ch_pick, ch_type in zip(ch_picks, ch_types):
            cols = [col for col in features_train.columns if ch_pick in col]
            cols = [
                col for col in cols if any(ch in col for ch in self.ch_names)
            ]
            if self.verbose:
                print("No. of features used:", len(cols))

            X_train = features_train[cols]
            X_test = features_test[cols]

            self.decoder.fit(X_train, y_train, groups_train)

            score = self.decoder.get_score(X_test, y_test)

            feature_importances = self._get_importances(
                feature_importance=self.feature_importance,
                decoder=self.decoder,
                data=X_test,
                label=y_test,
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
        self.fold += 1

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

        features_pred = self._get_feat_array_prediction(
            data=self.features[cols].values,
            events=self.events,
            events_used=events_used,
            sfreq=self.sfreq,
            begin=self.pred_begin,
            end=self.pred_end,
        )

        new_preds = (
            self._predict_epochs(
                self.decoder.model, features_pred, self.pred_mode
            )
            if len(features_pred) != 0
            else None
        )

        self.results.update_predictions(
            predictions_data=new_preds, ch_pick=ch_pick, ch_type=ch_type
        )

    @staticmethod
    def _get_importances(
        feature_importance: Union[int, bool],
        decoder: Decoder,
        data: np.ndarray,
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

    def _init_channel_names(
        self, ch_names: list, use_channels: str, side: str
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
        side = self._transform_side(side)
        if use_channels in case_contralateral:
            return [ch for ch in ch_names if side not in ch]
        if use_channels in case_ipsilateral:
            return [ch for ch in ch_names if side in ch]
        raise ValueError(
            f"Invalid argument for `use_channels`. Must be one of "
            f"{case_all+case_contralateral+case_ipsilateral}. Got: "
            f"{use_channels}."
        )

    @staticmethod
    def _transform_side(side: str) -> str:
        """Transform given keyword to a string for search."""
        if side == "right":
            return "R_"
        if side == "left":
            return "L_"
        raise ValueError(
            f"Invalid argument for `side`. Must be right "
            f"or left. Got: {side}."
        )

    def _get_ch_picks(self, features, y, groups, cv) -> list:
        """Handle channel picks."""
        if "single_best" in self.use_channels:
            return self._inner_loop(self.ch_names, features, y, groups, cv)
        if "all" in self.use_channels:
            return ["ECOG", "LFP"]
        return self.ch_names

    @staticmethod
    def _discard_trial(
        baseline: Union[int, float],
        data_artifacts: Optional[np.ndarray],
        index_epoch: int,
        bad_epochs: Iterable[int],
    ) -> bool:
        """"""
        if any(
            (
                baseline <= 0.0,
                np.count_nonzero(data_artifacts),
                index_epoch in bad_epochs,
            )
        ):
            return True
        return False

    def _get_feat_array(
        self,
        data: np.ndarray,
        events: Iterable,
        sfreq: Union[int, float],
        target_begin: Union[str, float, int] = "trial_onset",
        target_end: Union[str, float, int] = "trial_end",
        dist_onset: Union[float, int] = 2.0,
        dist_end: Union[float, int] = 2.0,
        artifacts: Optional[np.ndarray] = None,
        bad_epochs: Iterable[int] = None,
        verbose: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """"""
        dist_onset = int(dist_onset * sfreq)
        dist_end = int(dist_end * sfreq)

        rest_beg, rest_end = -5.0, -2.0
        rest_end_ind = int(rest_end * sfreq)
        target_begin = int(target_begin * sfreq)
        if target_end != "trial_end":
            target_end = int(target_end * sfreq)

        X, y, events_used, group_list, events_discard = [], [], [], [], []

        for i, ind in enumerate(np.arange(0, len(events), 2)):
            baseline_period = self._get_baseline_period(
                events, ind, dist_onset, dist_end, artifacts
            )
            rest_beg_ind = int(
                max(rest_end_ind - baseline_period, rest_beg * sfreq)
            )
            data_rest, data_target, data_art = self._get_trial_data(
                data,
                events,
                ind,
                target_begin,
                target_end,
                rest_beg_ind,
                rest_end_ind,
                artifacts,
            )
            if not self._discard_trial(
                baseline=baseline_period,
                data_artifacts=data_art,
                index_epoch=i,
                bad_epochs=bad_epochs,
            ):
                X.extend((data_rest, data_target))
                y.extend((np.zeros(len(data_rest)), np.ones(len(data_target))))
                events_used.append(ind)
                group_list.append(
                    np.full((len(data_rest) + len(data_target)), i)
                )
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

    @staticmethod
    def _get_feat_array_prediction(
        data: np.ndarray,
        events: Union[list, np.ndarray],
        events_used: np.ndarray,
        sfreq: Union[int, float],
        begin: Union[int, float],
        end: Union[int, float],
        verbose: bool = False,
    ) -> Optional[np.ndarray]:
        """"""
        begin = int(begin * sfreq)
        end = int(end * sfreq)
        epochs = []
        for ind in events_used:
            epoch = data[events[ind] + begin : events[ind] + end + 1]
            if len(epoch) == end - begin + 1:
                epochs.append(epoch.squeeze())
            else:
                if verbose:
                    print(
                        f"Mismatch of epoch samples. Got: {len(epoch)} "
                        f"samples. Expected: {end - begin + 1} samples. "
                        f"Discarding epoch: No. {ind + 1} of {len(events)}."
                    )
                else:
                    pass
        if epochs:
            return np.stack(epochs, axis=0)
        return np.array(epochs)

    def _get_baseline_period(
        self,
        events: np.ndarray,
        event_ind: int,
        dist_onset: int,
        dist_end: int,
        artifacts: Optional[np.ndarray],
    ) -> Union[int, float]:
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
                ind_art = bool_art[-1] if bool_art.size != 0 else 0.
                baseline = baseline - ind_art
        return baseline

    @staticmethod
    def _get_trial_data(
        data,
        events: np.ndarray,
        ind: int,
        target_begin: int,
        target_end: Union[int, str],
        rest_beg_ind: int,
        rest_end_ind: int,
        artifacts: Optional[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """"""
        data_art = None
        if target_end == "trial_end":
            data_rest = data[
                events[ind] + rest_beg_ind : events[ind] + rest_end_ind
            ]
            data_target = data[events[ind] + target_begin : events[ind + 1]]
            if artifacts is not None:
                data_art = artifacts[
                    events[ind] + target_begin : events[ind + 1]
                ]
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

    def _inner_loop(
        self,
        ch_names: list[Any],
        features: pd.DataFrame,
        labels: np.ndarray,
        groups: np.ndarray,
        cv=GroupShuffleSplit(n_splits=5, test_size=0.2),
    ) -> list[str, str]:
        """"""
        results = {ch_name: [] for ch_name in ch_names}
        for train_ind, test_ind in cv.split(features.values, labels, groups):
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
                X_train = features_train[cols].values
                X_test = features_test[cols].values
                self.decoder.fit(X_train, y_train, groups_train)
                y_pred = self.decoder.model.predict(X_test)
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

    @staticmethod
    def _predict_epochs(model, features: np.ndarray, mode: str) -> list[list]:
        """"""
        predictions = []
        if features.ndim < 3:
            np.expand_dims(features, axis=0)
        for trial in features:
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

    @staticmethod
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

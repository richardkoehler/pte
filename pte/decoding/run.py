"""Module for running a decoding experiment."""

import csv
import json
import os
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Union

import numpy as np
import pandas as pd
import sklearn
from matplotlib import pyplot as plt
from pte.decoding.decode_abc import Decoder
from sklearn.inspection import permutation_importance
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    LeaveOneGroupOut,
)


@dataclass
class Runner:
    """Class for running prediction experiments."""

    features: pd.DataFrame
    target_df: pd.DataFrame
    label_df: pd.DataFrame
    ch_names: list
    out_file: str
    decoder: Any
    artifacts: Optional[np.ndarray] = None
    bad_events: Optional[np.ndarray] = None
    sfreq: int = 10
    classifier: str = "lda"
    balancing: str = "oversample"
    scoring: str = "balanced_accuracy"
    optimize: bool = False
    target_begin: Union[str, float, int] = "trial_onset"
    target_end: Union[str, float, int] = "trial_end"
    dist_onset: Union[float, int] = 2.0
    dist_end: Union[float, int] = 2.0
    use_channels: str = "single"
    pred_begin: Union[float, int] = -3.0
    pred_end: Union[float, int] = 3.0
    pred_mode: str = "classify"
    cv_outer: sklearn.model_selection.BaseCrossValidator = GroupKFold(
        n_splits=5
    )
    cv_inner: sklearn.model_selection.BaseCrossValidator = GroupKFold(
        n_splits=5
    )
    show_plot: bool = False
    save_plot: str = False
    verbose: bool = False
    feature_importance: Any = False
    feature_epochs: pd.DataFrame = field(init=False)
    data_epochs: np.ndarray = field(init=False)
    fold: int = field(init=False)
    ch_picks: list = field(init=False)
    labels: np.ndarray = field(init=False)
    groups: np.ndarray = field(init=False)
    events_used: np.ndarray = field(init=False)
    events_discard: np.ndarray = field(init=False)
    evs_test: np.ndarray = field(init=False)
    predictions: dict = field(init=False)
    results: list = field(init=False)
    results_keys: list = field(init=False)
    features_dict: dict = field(init=False)
    target_name: str = field(init=False)
    label_name: str = field(init=False)
    events: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        # Initialize classification results
        if self.target_begin == "trial_onset":
            self.target_begin = 0.0
        if self.target_end == "trial_onset":
            self.target_end = 0.0
        self.side = "R_" if "R_" in self.out_file else "L_"
        self.ch_names = self._init_channel_names(
            self.ch_names, self.use_channels, self.side
        )
        self.target_name = self.target_df.columns[0]
        self.predictions = self._init_results(
            self.ch_names, self.use_channels, self.target_name
        )
        self.features_dict = self._init_results(
            self.ch_names, self.use_channels, self.target_name
        )
        self.label_name = self.label_df.name

        if self.bad_events is None:
            self.bad_events = np.atleast_1d([])

        self.predictions["Label"] = []
        self.predictions["LabelName"] = self.label_name
        self.features_dict["Label"] = []
        self.features_dict["LabelName"] = self.label_name
        self.features_dict["ChannelNames"] = self.ch_names

        # Calculate events from label
        self.events = _events_from_label(self.label_df.values, self.verbose)

        # Check for plausability of events
        if not (len(self.events) / 2).is_integer():
            raise ValueError(
                f"Number of events is odd. Found {len(self.events) / 2} events."
                f"Please check your data."
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
            bad_epochs=self.bad_events,
            verbose=self.verbose,
        )

        # Initialize DataFrame from array
        self.feature_epochs = pd.DataFrame(
            self.data_epochs, columns=self.features.columns
        )

        self.fold = 0
        self.results = []

    def run(self) -> None:
        """Calculate classification performance and out results."""

        print(
            f"Use channels: {self.use_channels}, Channels picked: {self.ch_names}"
        )
        # Outer cross-validation
        for train_ind, test_ind in self.cv_outer.split(
            self.data_epochs, self.labels, self.groups
        ):
            self._run_outer_cv(train_ind, test_ind)

        # Save results, check if directory exists
        if not os.path.isdir(os.path.dirname(self.out_file)):
            os.makedirs(os.path.dirname(self.out_file))
        if self.verbose:
            print("Writing results for file: ", self.out_file, "\n")

        # Plot Predictions
        if self.show_plot or self.save_plot:
            self._plot_predictions()

        # Save classification performance
        header = [
            "fold",
            "channel_name",
            self.scoring,
            "feature_importances",
            "trials_used",
            "trials_discarded",
            "IDs_discarded",
        ]
        with open(
            self.out_file + "_results.csv", "w", encoding="UTF8", newline=""
        ) as file:
            csvwriter = csv.writer(file, delimiter=",")
            csvwriter.writerow(header)
            csvwriter.writerows(self.results)

        # Save predictions time-locked to trial onset
        with open(
            self.out_file + "_predictions_timelocked.json",
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(self.predictions, file)

        # Save features time-locked to trial onset
        with open(
            self.out_file + "_features_timelocked.json", "w", encoding="utf-8"
        ) as file:
            json.dump(self.features_dict, file)

        # Save all features used for classificaiton
        self.feature_epochs["Label"] = self.labels
        self.feature_epochs.to_csv(
            self.out_file + "_features_concatenated.csv",
        )

    def _run_outer_cv(self, train_ind, test_ind):
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
        self.evs_test = np.unique(self.groups[test_ind]) * 2

        # Add label data to prediction results
        label_pred = self._get_feat_array_prediction(
            self.label_df.values,
            self.events,
            events_used=self.evs_test,
            sfreq=self.sfreq,
            begin=self.pred_begin,
            end=self.pred_end,
        )
        if label_pred.size > 0:
            self.predictions = self._add_label(
                predictions=self.predictions,
                label_name="Label",
                target_pred=label_pred,
            )
            self.features_dict = self._add_label(
                predictions=self.features_dict,
                label_name="Label",
                target_pred=label_pred,
            )

        # Add target data to prediction results
        target_pred = self._get_feat_array_prediction(
            self.target_df.values,
            self.events,
            events_used=self.evs_test,
            sfreq=self.sfreq,
            begin=self.pred_begin,
            end=self.pred_end,
            verbose=True,
        )
        if target_pred.size > 0:
            self.predictions = self._add_label(
                predictions=self.predictions,
                label_name="Target",
                target_pred=target_pred,
            )
            self.features_dict = self._add_label(
                predictions=self.features_dict,
                label_name="Target",
                target_pred=target_pred,
            )

        # Handle which channels are used
        ch_picks = self._get_ch_picks(
            features_train, y_train, groups_train, self.cv_inner
        )

        # Infer channel types
        ch_types = ["ECOG" if "ECOG" in ch else "LFP" for ch in ch_picks]

        # Perform classification for each selected model
        for ch_pick, ch_type in zip(ch_picks, ch_types):
            cols = [col for col in features_train.columns if ch_pick in col]
            if self.verbose:
                print("Channel: ", ch_pick)
                print("Number of features used: ", len(cols))

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

            # Add results to list
            self.results.append(
                [
                    self.fold,
                    ch_pick,
                    score,
                    feature_importances,
                    len(self.events_used),
                    len(self.events) // 2 - len(self.events_used),
                    self.events_discard,
                ]
            )

            features_pred = self._get_feat_array_prediction(
                data=self.features[cols].values,
                events=self.events,
                events_used=self.evs_test,
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

            self.predictions = self._append_results(
                results=self.predictions,
                new_preds=new_preds,
                use_channels=self.use_channels,
                ch_pick=ch_pick,
                ch_type=ch_type,
            )

            self.features_dict = self._append_results(
                results=self.features_dict,
                new_preds=features_pred,
                use_channels=self.use_channels,
                ch_pick=ch_pick,
                ch_type=ch_type,
            )

        self.fold += 1

    @staticmethod
    def _append_results(
        results: dict,
        new_preds: list,
        use_channels: str,
        ch_pick: str,
        ch_type: str,
    ) -> dict:
        """Append new results to existing results dictionary."""
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

    def _plot_predictions(self):
        """Plot predictions."""
        for ch_name in self.predictions.keys():
            title = (
                ch_name
                + ": Classification target "
                + str(self.target_begin)
                + " - "
                + str(self.target_end)
            )
            self._plot_channel_predictions(
                self.predictions[ch_name],
                label=self.predictions["Movement"],
                label_name="Movement",
                title=title,
                sfreq=self.sfreq,
                axis_time=(self.pred_begin, self.pred_end),
                savefig=self.save_plot,
                show_plot=self.show_plot,
                filename=self.out_file,
            )

    @staticmethod
    def _add_label(
        predictions: dict, label_name: str, target_pred: np.ndarray
    ) -> dict:
        """Append array of labels to classifications results."""
        if target_pred.ndim == 1:
            target_pred = np.expand_dims(target_pred, axis=0)
        for i, epoch in enumerate(target_pred):
            # Invert array if necessary
            if abs(epoch.min()) > abs(epoch.max()):
                target_pred[i] = epoch * -1.0
            # Perform min-max scaling
            target_pred[i] = (epoch - epoch.min()) / (
                epoch.max() - epoch.min()
            )
        predictions[label_name].extend(target_pred.tolist())
        return predictions

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
        else:
            raise ValueError(
                f"`feature_importances` must be an integer or `False`. Got: {feature_importance}."
            )

    @staticmethod
    def _init_channel_names(
        ch_names: list, use_channels: str, side: str
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
        if use_channels in case_contralateral:
            return [ch for ch in ch_names if side not in ch]
        if use_channels in case_ipsilateral:
            return [ch for ch in ch_names if side in ch]
        raise ValueError(
            f"Got false value for `use_channels`. Must be one of "
            f"{case_all+case_contralateral+case_ipsilateral}. Got: {use_channels}."
        )

    @staticmethod
    def _init_results(
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
                f"Input value for `use_channels` not allowed. Got: {use_channels}."
            )
        return results

    def _get_ch_picks(self, features, y, groups, cv) -> list:
        """Handle channel picks."""
        if "single_best" in self.use_channels:
            return self._inner_loop(self.ch_names, features, y, groups, cv)
        if "all" in self.use_channels:
            return ["ECOG", "LFP"]
        return self.ch_names

    @staticmethod
    def _discard_trial(
        baseline: int,
        data_artifacts: Optional[np.ndarray],
        index_epoch: int,
        bad_epochs: Union[np.ndarray, list],
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
        data,
        events,
        sfreq,
        target_begin,
        target_end,
        dist_onset,
        dist_end,
        artifacts=None,
        bad_epochs=None,
        verbose=False,
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
        events: Iterable,
        events_used: Iterable,
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
                        f"Mismatch of epoch samples. Got: {len(epoch)} samples, expected: "
                        f"{end - begin + 1} samples. Epoch: No. {ind + 1} of "
                        f"{len(events)}. Discarding epoch."
                    )
                else:
                    pass
        if epochs:
            return np.stack(epochs, axis=0)
        return np.array(epochs)

    def _get_baseline_period(
        self,
        events,
        event_ind: int,
        dist_onset: int,
        dist_end: int,
        artifacts: Optional[np.ndarray],
    ) -> int:
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

    @staticmethod
    def _get_trial_data(
        data,
        events,
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
                    f"Only `classification`, `probability` or `decision_function` "
                    f"are valid options for `mode`. Got {mode}."
                )
            predictions.append(pred)
        return predictions

    @staticmethod
    def _plot_channel_predictions(
        predictions,
        label,
        label_name: str,
        title: str,
        sfreq: Union[int, str],
        axis_time: tuple[Union[int, str], Union[int, str]],
        savefig=False,
        show_plot=False,
        filename=None,
    ) -> None:
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


def _events_from_label(
    label_data: np.ndarray, verbose: bool = False
) -> np.ndarray:
    """

    Parameters
    ----------
    label_data
    verbose

    Returns
    -------
    events
    """
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

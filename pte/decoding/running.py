"""Module for running a decoding experiment."""

import csv
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Union

import numpy as np
import pandas as pd
import sklearn
from matplotlib import pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.metrics import balanced_accuracy_score, log_loss
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    LeaveOneGroupOut,
)

from ..settings import PATH_PYNEUROMODULATION
from .decode import get_decoder

sys.path.insert(0, PATH_PYNEUROMODULATION)

from pyneuromodulation.nm_reader import NM_Reader


def run_prediction(
    features_root,
    feature_file,
    classifier,
    label_channels,
    target_begin,
    target_end,
    optimize,
    balancing,
    out_root,
    use_channels,
    use_features,
    cross_validation,
    scoring="balanced_accuracy",
    feature_importance=False,
    plot_target_channels=None,
    pred_mode="classify",
    use_times=1,
    exceptions=None,
    save_plot=True,
    show_plot=False,
    verbose=True,
) -> None:
    """Initialize Runner object and save prediction and save results."""

    if verbose:
        print("Using file: ", feature_file)

    nm_reader = NM_Reader(feature_path=features_root)
    features = nm_reader.read_features(feature_file)
    settings = nm_reader.read_settings(feature_file)

    # Pick label for classification
    for label_channel in label_channels:
        if label_channel in features.columns:
            label = nm_reader.read_label(label_channel)
            break

    # Pick artifact channel
    artifacts = []
    # artifacts = file_reader.read_label(ARTIFACT_CHANNELS)

    # Calculate events from label
    events = _events_from_label(label.values, verbose)

    # Pick target for plotting predictions
    target_df = _get_target_df(plot_target_channels, features)

    features_df = _get_feature_df(features, use_features, use_times)

    # Generate output file name
    out_path = _generate_outpath(
        out_root,
        feature_file,
        classifier,
        target_begin,
        target_end,
        use_channels,
        optimize,
        use_times,
    )

    decoder = get_decoder(
        classifier=classifier,
        scoring=scoring,
        balancing=balancing,
        optimize=optimize,
    )

    # Initialize Runner instance
    runner = Runner(
        features=features_df,
        target=target_df,
        events=events,
        artifacts=artifacts,
        ch_names=settings["ch_names"],
        sfreq=settings["sampling_rate_features"],
        classifier=classifier,
        balancing=balancing,
        optimize=optimize,
        decoder=decoder,
        target_begin=target_begin,
        target_end=target_end,
        dist_onset=2.0,
        dist_end=1.5,
        exception_files=exceptions,
        excep_dist_end=0.5,
        use_channels=use_channels,
        pred_begin=-3.0,
        pred_end=2.0,
        pred_mode=pred_mode,
        cv_outer=cross_validation,
        feature_importance=feature_importance,
        show_plot=show_plot,
        save_plot=save_plot,
        out_file=out_path,
        verbose=verbose,
    )
    runner.run()


@dataclass
class Runner:
    """Class for running prediction experiments."""

    features: pd.DataFrame
    target: pd.DataFrame
    events: np.ndarray
    artifacts: np.ndarray
    ch_names: list
    out_file: str
    decoder: Any
    sfreq: int = 10
    classifier: str = "lda"
    balancing: str = "oversample"
    scoring: str = "balanced_accuracy"
    optimize: bool = False
    target_begin: Union[str, float, int] = "MovementOnset"
    target_end: Union[str, float, int] = "MovementEnd"
    dist_onset: Union[float, int] = 2.0
    dist_end: Union[float, int] = 2.0
    exception_files: Optional[list] = None
    excep_dist_end: Union[float, int] = 0.0
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

    def __post_init__(self) -> None:
        # Initialize classification results
        self.side = "R_" if "R_" in self.out_file else "R_"
        self.ch_names = self._init_channel_names(
            self.ch_names, self.use_channels, self.side
        )
        self.predictions = self._init_results(self.ch_names, self.use_channels)
        self.features_dict = self._init_results(
            self.ch_names, self.use_channels
        )
        self.dist_end = self._handle_exception_files()
        self.fold = 0
        self.results = []

    def run(self) -> None:
        """Calculate classification performance and write to *.tsv file."""

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
            verbose=self.verbose,
        )

        # Initialize DataFrame from array
        self.feature_epochs = pd.DataFrame(
            self.data_epochs, columns=self.features.columns
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

        # Save predictions
        with open(self.out_file + "_predictions.json", "w") as fp:
            json.dump(self.predictions, fp, indent=4)
        # classif_df = pd.DataFrame.from_dict(data=self.classifications)
        # classif_df.to_csv(self.out_file + "_classif.tsv", sep="\t")

        # Save features
        with open(self.out_file + "_features.json", "w") as fp:
            json.dump(self.features_dict, fp, indent=4)

    def _run_outer_cv(self, train_ind, test_ind):
        if self.verbose:
            print(f"Fold no.: {self.fold}")

        # Get training and testing data and labels
        self.features_train, self.features_test = (
            self.feature_epochs.iloc[train_ind],
            self.feature_epochs.iloc[test_ind],
        )
        self.y_train = np.ascontiguousarray(self.labels[train_ind])
        self.y_test = np.ascontiguousarray(self.labels[test_ind])
        self.groups_train = self.groups[train_ind]

        # Get prediction epochs
        self.evs_test = np.unique(self.groups[test_ind]) * 2

        # Add target data to classifications results
        self._add_target_array()

        # Handle which channels are used
        ch_picks = self._get_ch_picks()

        # Infer channel types
        ch_types = ["ECOG" if "ECOG" in ch else "LFP" for ch in ch_picks]

        # Perform classification for each selected model
        for ch_pick, ch_type in zip(ch_picks, ch_types):
            cols = [
                col for col in self.features_train.columns if ch_pick in col
            ]
            if self.verbose:
                print("Channel: ", ch_pick)
                print("Number of features used: ", len(cols))

            X_train = np.ascontiguousarray(self.features_train[cols].values)
            X_test = np.ascontiguousarray(self.features_test[cols].values)

            self.decoder.fit(X_train, self.y_train, self.groups_train)

            feature_importances = self._get_importances(
                feature_importance=self.feature_importance,
                model=self.decoder.model,
                data=X_test,
                label=self.y_test,
                scoring=self.scoring,
            )

            score = self.decoder.get_score(X_test, self.y_test)

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
                if features_pred is not None
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

    def _make_predictions(
        self,
        model,
        pred_mode,
        features,
        events,
        events_test,
        sfreq,
        pred_begin,
        pred_end,
    ) -> Optional[list]:
        """Make predictions for events based on given features."""
        features_pred = self._get_feat_array_prediction(
            features,
            events,
            events_test,
            sfreq=sfreq,
            begin=pred_begin,
            end=pred_end,
        )
        if len(features_pred) == 0:
            return None
        return self._predict_epochs(model, features_pred, pred_mode)

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

    def _add_target_array(self) -> None:
        """Append target array to classifications results."""

        target_pred = self._get_feat_array_prediction(
            self.target.values,
            self.events,
            events_used=self.evs_test,
            sfreq=self.sfreq,
            begin=self.pred_begin,
            end=self.pred_end,
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
            self.predictions["Movement"].extend(target_pred.tolist())

    @staticmethod
    def _get_importances(
        feature_importance, model, data, label, scoring
    ) -> list:
        """Calculate feature importances."""
        if not feature_importance:
            return []
        if isinstance(feature_importance, int):
            imp_scores = permutation_importance(
                model,
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
        # imp_scores = model.coef_

    @staticmethod
    def _init_channel_names(
        ch_names: list, use_channels: str, side: str
    ) -> list:
        """Initialize channels to be used."""
        if use_channels in ["single", "single_best", "all"]:
            return ch_names
        if use_channels in ["single_contralat", "all_contralat"]:
            return [ch for ch in ch_names if side not in ch]
        if use_channels in ["single_ipsilat", "all_ipsilat"]:
            return [ch for ch in ch_names if side in ch]

    @staticmethod
    def _init_results(ch_names: list, use_channels: str) -> dict:
        """Initialize results dictionary."""
        results = {}
        results.update({"Movement": []})
        if use_channels in [
            "all",
            "all_contralat",
            "all_ipsilat",
            "single_best",
        ]:
            results.update({ch: [] for ch in ["ECOG", "LFP"]})
        elif use_channels in ["single", "single_contralat", "single_ipsilat"]:
            results.update({ch: [] for ch in ch_names})
        else:
            raise ValueError(
                f"Input `use_channels` not valid. Got: {use_channels}."
            )
        return results

    def _handle_exception_files(self):
        """Check if current file is listed in exception files."""
        if all(
            (
                self.exception_files,
                any([exc in self.out_file for exc in self.exception_files]),
            )
        ):
            print(
                "Exception file recognized: ", os.path.basename(self.out_file)
            )
            return self.excep_dist_end
        return self.dist_end

    def _get_ch_picks(self) -> list:
        """Handle channel picks."""
        picks = {
            "single": self.ch_names,
            "single_contralat": [
                ch for ch in self.ch_names if self.side not in ch
            ],
            "single_ipsilat": [ch for ch in self.ch_names if self.side in ch],
            # "single_best": sorted(
            #     self._inner_loop(
            #         self.ch_names,
            #         self.features_train,
            #         self.y_train,
            #         self.groups_train,
            #         self.cv_inner,
            #     )
            # ),
            "all": ["ECOG", "LFP"],
            "all_ipsilat": (
                ["ECOG", "LFP_L"] if self.side == "L_" else ["ECOG", "LFP_R"]
            ),
            "all_contralat": (
                ["ECOG", "LFP_R"] if self.side == "L_" else ["ECOG", "LFP_L"]
            ),
        }
        if self.use_channels not in picks:
            raise ValueError(
                f"use_channels keyword not valid. Must be one of {picks.keys}: "
                f"{self.use_channels}"
            )
        return picks[self.use_channels]

    @staticmethod
    def _discard_trial(baseline: int, data_artifacts: np.ndarray) -> bool:
        """"""
        if any((baseline <= 0.0, np.count_nonzero(data_artifacts))):
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
        verbose=False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
            if not self._discard_trial(baseline_period, data_art):
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
        data, events, events_used, sfreq, begin, end
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
                print(
                    f"Length mismatch of epochs. Got: {len(epoch)}, expected: "
                    f"{end - begin + 1}. Event used: No. {ind + 1} of "
                    f"{len(events)}."
                )
        if epochs:
            return np.stack(epochs, axis=0)
        return

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
        artifacts,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """"""
        data_art = None
        if target_end == "MovementEnd":
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
            y_train, y_test = (
                np.ascontiguousarray(labels[train_ind]),
                np.ascontiguousarray(labels[test_ind]),
            )
            groups_train = groups[train_ind]
            for ch_name in ch_names:
                cols = [
                    col for col in features_train.columns if ch_name in col
                ]
                X_train = np.ascontiguousarray(features_train[cols].values)
                X_test = np.ascontiguousarray(features_test[cols].values)
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


def _generate_outpath(
    root: str,
    feature_file: str,
    classifier: str,
    target_begin: Union[str, int, float],
    target_end: Union[str, int, float],
    use_channels: str,
    optimize: bool,
    use_times: int,
) -> str:
    """Generate file name for output files."""
    clf_str = "_" + classifier + "_"

    target_str = (
        "movement_"
        if target_end == "MovementEnd"
        else "mot_intention_" + str(target_begin) + "_" + str(target_end) + "_"
    )
    ch_str = use_channels + "_chs_"
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

    """
    label_diff = np.zeros_like(label_data, dtype=int)
    label_diff[1:] = np.diff(label_data)
    events_ = np.nonzero(label_diff)[0]
    if verbose:
        print(f"Number of events detected: {len(events_) / 2}")
    return events_


def _get_target_df(
    targets: Iterable, features_df: pd.DataFrame, verbose: bool = False
) -> pd.DataFrame:
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
    if verbose:
        print("Target channel used: ", target_df.columns[0])
    return target_df


def _get_feature_df(
    features: pd.DataFrame, use_features: Iterable, use_times: int
) -> pd.DataFrame:
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

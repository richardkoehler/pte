"""Module for running a decoding experiment."""

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
    target_beg,
    target_en,
    optimize,
    balance,
    out_root,
    use_channels,
    use_features,
    plot_target_channels=None,
    pred_mode="probability",
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
    channels = np.array(settings["ch_names"])[settings["feature_idx"]]

    # Pick label for classification
    for label_channel in label_channels:
        if label_channel in features.columns:
            label = nm_reader.read_label(label_channel)
            if verbose:
                print(f"Using label: {label_channel}")
            break

    # Pick artifact channel
    artifacts = []
    # artifacts_ = file_reader.read_label(ARTIFACT_CHANNELS)

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
        target_beg,
        target_en,
        use_channels,
        optimize,
        use_times,
    )

    runner = Runner(
        features=features_df,
        target=target_df,
        events=events,
        artifacts=artifacts,
        ch_names=channels,
        out_file=out_path,
        sfreq=settings["sampling_rate_features"],
        classifier=classifier,
        balancing=balance,
        optimize=optimize,
        target_begin=target_beg,
        target_end=target_en,
        dist_onset=2.0,
        dist_end=1.5,
        exception_files=exceptions,
        excep_dist_end=0.5,
        use_channels=use_channels,
        pred_begin=-3.0,
        pred_end=2.0,
        pred_mode=pred_mode,
        cv_outer=LeaveOneGroupOut(),
        show_plot=show_plot,
        save_plot=save_plot,
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
    feature_epochs: pd.DataFrame = field(init=False)
    classifications: pd.DataFrame = field(init=False)
    ch_picks: list = field(init=False)
    data_epochs: np.ndarray = field(init=False)
    labels: np.ndarray = field(init=False)
    events_used: np.ndarray = field(init=False)
    groups: np.ndarray = field(init=False)
    events_discard: np.ndarray = field(init=False)
    fold: int = field(init=False)
    evs_test: np.ndarray = field(init=False)
    results: list = field(init=False)
    results_keys: list = field(init=False)

    def __post_init__(self):
        # Initialize classification results
        self.classifications = self._init_classifications()
        self.dist_end = self._handle_exception_files()
        self.fold = 0
        self.results = []

    def run(self):
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
        results_df = pd.DataFrame(
            data=self.results,
            columns=[
                self.scoring,
                "fold",
                "channel_name",
                "feature_importances",
                "trials_used",
                "trials_discarded",
                "IDs_discarded",
            ],
        )
        results_df.to_csv(self.out_file + "_results.tsv", sep="\t")

        # Save predictions
        classif_df = pd.DataFrame.from_dict(data=self.classifications)
        classif_df.to_csv(self.out_file + "_classif.tsv", sep="\t")

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
            decoder = get_decoder(
                classifier=self.classifier,
                scoring=self.scoring,
                balancing=self.balancing,
                optimize=self.optimize,
            )
            decoder.fit(X_train, self.y_train, self.groups_train)
            imp = permutation_importance(
                decoder.model,
                X_test,
                self.y_test,
                scoring=self.scoring,
                n_repeats=100,
                n_jobs=-1,
            )
            imp_scores = imp.importances_mean
            # imp_scores = model.coef_
            score = decoder.get_score(X_test, self.y_test)
            self.results.append(
                [
                    score,
                    self.fold,
                    ch_pick,
                    imp_scores,
                    len(self.events_used),
                    len(self.events) // 2 - len(self.events_used),
                    self.events_discard,
                ]
            )

            # Perform predictions
            features_pred = self._get_feat_array_prediction(
                self.features[cols].values,
                self.events,
                self.evs_test,
                sfreq=self.sfreq,
                begin=self.pred_begin,
                end=self.pred_end,
            )
            if len(features_pred) == 0:
                pass
            else:
                # Make predictions
                predictions = self._predict_epochs(
                    decoder.model, features_pred, self.pred_mode
                )
                # Add prediction results to dictionary
                if self.use_channels in ["single", "single_contralat"]:
                    self.classifications[ch_pick].extend(predictions)
                else:
                    self.classifications[ch_type].extend(predictions)
        self.fold += 1

    def _plot_predictions(self):
        """Plot predictions."""
        for ch_name in self.classifications.keys():
            title = (
                ch_name
                + ": Classification target "
                + str(self.target_begin)
                + " - "
                + str(self.target_end)
            )
            self._plot_channel_predictions(
                self.classifications[ch_name],
                label=self.classifications["Movement"],
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
            self.classifications["Movement"].extend(target_pred)

    def _init_classifications(self):
        """Initialize classification results dictionary."""
        classifications = {}
        classifications.update({"Movement": []})
        if self.use_channels == "single":
            classifications.update({ch: [] for ch in self.ch_names})
        elif self.use_channels == "single_contralat":
            side = "L_" if "R_" in self.out_file else "R_"
            ch_names = [ch for ch in self.ch_names if side in ch]
            classifications.update({ch: [] for ch in ch_names})
        else:
            classifications.update({ch: [] for ch in ["ECOG", "LFP"]})
        return classifications

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
            "single_contralat": self.ch_names,
            "single_best": sorted(
                self._inner_loop(
                    self.ch_names,
                    self.features_train,
                    self.y_train,
                    self.groups_train,
                    self.cv_inner,
                )
            ),
            "all": ["ECOG", "LFP"],
            "all_ipsilat": (
                ["ECOG", "LFP_L"]
                if "L_" in self.out_file
                else ["ECOG", "LFP_R"]
            ),
            "all_contralat": (
                ["ECOG", "LFP_R"]
                if "L_" in self.out_file
                else ["ECOG", "LFP_L"]
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
    ) -> tuple(np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray):
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

    def _get_feat_array_prediction(
        self, data, events, events_used, sfreq, begin, end
    ) -> Union[str, np.ndarray]:
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
        return epochs

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

    def _get_trial_data(
        self,
        data,
        events,
        ind: int,
        target_begin: int,
        target_end: Union[int, str],
        rest_beg_ind: int,
        rest_end_ind: int,
        artifacts,
    ) -> tuple(np.ndarray, np.ndarray, np.ndarray):
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
        ch_names: list(Any),
        features: pd.DataFrame,
        labels: np.ndarray,
        groups: np.ndarray,
        cv=GroupShuffleSplit(n_splits=5, test_size=0.2),
    ) -> list(str, str):
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
                decoder = get_decoder(
                    self.classifier, self.balancing, self.optimize
                )
                decoder.fit(X_train, y_train, groups_train)
                y_pred = decoder.model.predict(X_test)
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
    def _predict_epochs(model, features: np.ndarray, mode: str) -> list(Any):
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

    @staticmethod
    def _plot_channel_predictions(
        predictions,
        label,
        label_name: str,
        title: str,
        sfreq: Union[int, str],
        axis_time: Iterable[Union[int, str], Union[int, str]],
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
    root,
    feature_file,
    classifier,
    target_beg,
    target_en,
    use_channels_,
    optimize,
    use_times,
) -> str:
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
    targets: Iterable, features_df: pd.DataFrame
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
    print("Target channel used: ", target_df.columns)
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

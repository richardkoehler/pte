import json
import os

import pandas as pd

import mne_bids


ecog_hemisphere = {
    '002': 'L',
    '003': 'L',
    '004': 'L',
    '005': 'R',
    '006': 'R',
    '007': 'R',
    'FOG006': 'R',
    'FOG008': 'R',
    'FOG010': 'R',
    'FOG011': 'R',
    'FOG012': 'R',
    'FOG014': 'R',
    'FOG016': 'R',
    'FOG021': 'R',
    'FOG022': 'R',
    'FOGC001': 'R'
}


class FileReader:

    def __init__(self, root, verbose=True) -> None:

        self.root = root
        self.files = None
        self.df_M1 = None
        self.settings = None
        self.features = None
        self.feature_ch = None
        self.run_analysis = None
        self.verbose = verbose

    def get_all_files(self, path, suffix, get_bids=False, prefix=None,
                      extension=None):
        """Return all files in (sub-)directories of path with given
        suffixes and prefixes (case-insensitive).

        Args:
            path (string)
            suffix (iterable): e.g. ["vhdr", "edf"] or ".json"
            get_bids (boolean): True if BIDS_Path type should be returned
            instead of string. Default: False
            bids_root (string/path): Path of BIDS root folder. Only required if
            get_bids=True.
            prefix (iterable): e.g. ["SelfpacedRota", "ButtonPress] (optional)
            extension (string): any keyword to search for in filename
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
                    if file.lower().endswith(suff.lower()):
                        if not prefix:
                            filepaths.append(os.path.join(root, file))
                        else:
                            for pref in prefix:
                                if pref.lower() in file.lower():
                                    filepaths.append(os.path.join(root, file))
        if get_bids:
            filepaths = self.__make_bids_paths(filepaths, '.vhdr')
        self.files = filepaths
        if self.verbose:
            self.print_files(self.files)
        return self.files

    def get_bids_files(self) -> list:
        self.get_all_files(
            path=self.root, suffix=['.vhdr', '.edf'], get_bids=True,
            prefix=None, extension=None)
        return self.files

    def filter_files(self, keywords=None, hemisphere=None, stimulation=None,
                     medication=None, exclude=None) -> list:
        is_bids = False
        if isinstance(self.files[0], mne_bids.path.BIDSPath):
            filtered_files = [file.basename for file in self.files]
            is_bids = True
        else:
            filtered_files = self.files
        if keywords:
            if isinstance(keywords, str):
                keywords = [keywords]
            filtered_files = [
                file for file in filtered_files if any(
                    [key in file for key in keywords])]
        if stimulation:
            if stimulation.lower() in 'stimon':
                stim = 'StimOn'
            elif stimulation.lower() in 'stimoff':
                stim = 'StimOff'
            else:
                raise ValueError('Keyword for stimulation not valid.')
            filtered_files = [file for file in filtered_files if stim in file]
        if medication:
            if medication.lower() in 'medon':
                med = 'MedOn'
            elif medication.lower() in 'medoff':
                med = 'MedOff'
            else:
                raise ValueError('Keyword for medication not valid.')
            filtered_files = [file for file in filtered_files if med in file]
        if hemisphere:
            matching_files = []
            for file in filtered_files:
                entities = mne_bids.get_entities_from_fname(file)
                hem = ecog_hemisphere[entities['subject']] + '_'
                if hemisphere.lower() in 'ipsilateral' \
                        and hem in file:
                    matching_files.append(file)
                if hemisphere.lower() in 'contralateral' \
                        and hem not in file:
                    matching_files.append(file)
            filtered_files = matching_files
        if exclude:
            filtered_files = [file for file in filtered_files if not any(
                item in file for item in exclude)]
        if self.verbose:
            self.print_files(filtered_files)
        if is_bids:
            filtered_files = self.__make_bids_paths(filtered_files, '.vhdr')
        return filtered_files

    def get_feature_list(self) -> list:
        f_files = []
        for dirpath, subdirs, files in os.walk(self.root):
            for x in files:
                if "FEATURES" in x:
                    f_files.append(os.path.basename(dirpath))
        return f_files

    def __make_bids_paths(self, filepaths, extension) -> list:
        if not self.root:
            print(
                "Warning: No root folder given. Please pass bids_root "
                "parameter to create a complete BIDS_Path object.")
        bids_paths = []
        for filepath in filepaths:
            entities = mne_bids.get_entities_from_fname(filepath)
            try:
                bids_path = mne_bids.BIDSPath(
                    subject=entities["subject"], session=entities["session"],
                    task=entities["task"], run=entities["run"],
                    acquisition=entities["acquisition"],
                    suffix=entities["suffix"], extension=extension,
                    root=self.root)
            except ValueError as err:
                print(f"ValueError while creating BIDS_Path object for file "
                      f"{filepath}: {err}")
            else:
                bids_paths.append(bids_path)
        return bids_paths

    def print_files(self, files) -> None:
        if not files:
            print("No corresponding files found.")
        else:
            print('Corresponding files found:')
            for idx, file in enumerate(files):
                print(idx, ':', os.path.basename(file))

    def read_settings(self, feature_file) -> None:
        self.feature_file = feature_file
        with open(os.path.join(self.root, feature_file,
                               feature_file + "_SETTINGS.json")) as f:
            self.settings = json.load(f)
        return self.settings

    def read_M1(self, feature_file) -> None:
        self.df_M1 = pd.read_csv(os.path.join(self.root, feature_file,
                                              feature_file + "_DF_M1.csv"),
                                 header=0)
        return self.df_M1

    def read_features(self, feature_file) -> None:
        self.features = pd.read_csv(
            os.path.join(self.root, feature_file,
                         feature_file + "_FEATURES.csv"), header=0)
        return self.features

    def read_channel_data(self, ch_name, read_bp_activity_only=False,
                          read_sharpwave_prominence_only=False) -> None:
        self.ch_name = ch_name
        self.feature_ch_cols_all = [i for i in list(self.features.columns) if
                                    ch_name in i]
        if read_bp_activity_only:
            bp_ = [f for f in self.feature_ch_cols_all if all(
                x in f for x in ('bandpass', 'activity'))]
            self.feature_ch_cols = bp_[::-1]  # flip list s.t. theta band is lowest in subsequent plot
        elif read_sharpwave_prominence_only is True:
            self.feature_ch_cols = [
                f for f in self.feature_ch_cols_all if all(
                    x in f for x in ('Sharpwave', 'prominence'))]
        else:
            self.feature_ch_cols = self.feature_ch_cols_all
        self.feature_ch = self.features[self.feature_ch_cols]
        return self.feature_ch

    def read_label(self, label_name) -> None:
        self.label_name = label_name
        self.label = self.features[label_name]
        return self.label

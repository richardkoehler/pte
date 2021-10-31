"""Module for handling datasets in BIDS-format."""

import os
import shutil
from typing import List

import mne
import mne_bids
import numpy as np
import pandas as pd
import pybv


def add_coord_column(
    df_chs: pd.DataFrame, ch_names: List[str], new_ch: str
) -> pd.DataFrame:
    """Add column by interpolating coordinates from given channels.

    Arguments
    ---------
    df_chs : pd.DataFrame
        DataFrame as read in from a BIDS-compliant *electrodes.tsv file.
    ch_names : list of str
        Channels to use for interpolating coordinates for new channel.
    new_ch : str
        Channel name of new channel.
    
    Returns
    -------
    df_chs : pd.DataFrame
        New DataFrame with appended column.
    """
    coords = [
        df_chs.loc[ch_name][["x", "y", "z"]].values for ch_name in ch_names
    ]
    coords = np.array(coords, float).mean(axis=0).round(5)
    append = df_chs.loc[ch_names[0]].copy()
    append.loc[["x", "y", "z"]] = coords
    append.name = new_ch
    df_chs = df_chs.append(append, ignore_index=False)
    df_chs.sort_index(axis=0, inplace=True)
    return df_chs


def bids_save_file(
    raw: mne.io.Raw, bids_path: mne_bids.BIDSPath
) -> mne.io.Raw:
    """Write preloaded data to BrainVision file in BIDS format.

    Parameters
    ----------
    raw : raw MNE object
        The raw MNE object for this function to write
    bids_path : BIDSPath MNE-BIDS object
        The MNE BIDSPath to the file to be overwritten
    return_raw : boolean, optional
        Set to True to return the new raw object that has been written.
        Default is False.
    Returns
    -------
    raw : raw MNE object or None
        The newly written raw object.
    """

    data = raw.get_data()
    sfreq = raw.info["sfreq"]
    ch_names = raw.ch_names
    folder = bids_path.directory
    # events, event_id = mne.events_from_annotations(raw)

    # rewrite datafile
    pybv.write_brainvision(
        data=data,
        sfreq=sfreq,
        ch_names=ch_names,
        events=None,
        fname_base="dummy",
        folder_out=folder,
    )
    source_path = os.path.join(folder, "dummy" + ".vhdr")
    raw = mne.io.read_raw_brainvision(source_path)
    raw.info["line_freq"] = 50

    mapping_dict = _get_mapping_dict(raw.ch_names)
    raw.set_channel_types(mapping_dict)

    mne_bids.write_raw_bids(raw, bids_path, overwrite=True)
    suffixes = [".eeg", ".vhdr", ".vmrk"]
    dummy_files = [
        os.path.join(folder, "dummy" + suffix) for suffix in suffixes
    ]
    for dummy_file in dummy_files:
        os.remove(dummy_file)
    # check for success
    raw = mne_bids.read_raw_bids(bids_path, verbose=False)
    return raw


def _get_mapping_dict(ch_names: List[str]) -> dict:
    """Create dictionary for remapping channel types.
    
    Arguments
    ---------
    ch_names : list
        List of channel names to be remapped.
    
    Returns
    -------
    remapping_dict : dict
        Dictionary mapping each channel name to a channel type.
    """
    remapping_dict = {}
    for ch_name in ch_names:
        if ch_name.startswith("ECOG"):
            remapping_dict[ch_name] = "ecog"
        elif ch_name.startswith(("LFP", "STN")):
            remapping_dict[ch_name] = "dbs"
        elif ch_name.startswith("EMG"):
            remapping_dict[ch_name] = "emg"
        elif ch_name.startswith("EEG"):
            remapping_dict[ch_name] = "eeg"
        elif ch_name.startswith(
            ("MOV", "ANALOG", "ROT", "ACC", "AUX", "X", "Y", "Z", "MISC")
        ):
            remapping_dict[ch_name] = "misc"
        else:
            remapping_dict[ch_name] = "misc"
    return remapping_dict


def bids_rewrite_file(
    raw: mne.io.Raw, bids_path: mne_bids.BIDSPath
) -> mne.io.Raw:
    """Overwrite BrainVision data in BIDS format that has been modified.

    Parameters
    ----------
    raw : raw MNE object
        The raw MNE object for this function to write
    bids_path : BIDSPath MNE-BIDS object
        The MNE BIDSPath to the file to be overwritten
   
    Returns
    -------
    raw_new : raw MNE object
        The newly written raw object.
    """
    curr_path = bids_path.copy().update(
        suffix=bids_path.datatype, extension=None
    )
    raw_copy = raw.copy()
    working_dir = curr_path.directory

    temp_root = os.path.join(working_dir, "temp")
    if not os.path.isdir(temp_root):
        os.mkdir(temp_root)
    temp_path = curr_path.copy().update(root=temp_root)
    raw_copy.info["dig"] = None
    mne_bids.write_raw_bids(
        raw_copy,
        temp_path,
        allow_preload=True,
        format="BrainVision",
        verbose=False,
    )

    # Rewrite BrainVision files
    mne_bids.copyfiles.copyfile_brainvision(temp_path, curr_path.fpath)

    _rewrite_events(temp_path, curr_path)

    # Rewrite **channels.tsv
    channels_path = bids_path.copy().update(
        suffix="channels", extension=".tsv"
    )
    df_chs = pd.read_csv(channels_path.fpath, sep="\t", index_col=0)
    old_chs = df_chs.index.tolist()
    add_chs = [ch for ch in raw.ch_names if ch not in old_chs]
    description = {
        "seeg": "StereoEEG",
        "ecog": "Electrocorticography",
        "eeg": "Electroencephalography",
        "emg": "Electromyography",
        "misc": "Miscellaneous",
        "dbs": "Deep Brain Stimulation",
    }
    if add_chs:
        add_list = []
        ch_types = raw_copy.get_channel_types(picks=add_chs)
        print("Added channels: ", add_chs)
        print("Added channel types: ", ch_types)
        for idx in len(add_chs):
            add_dict = {
                df_chs.columns[i]: df_chs.iloc[0][i]
                for i in range(0, len(df_chs.columns))
            }
            add_dict.update({"type": ch_types[idx].upper()})
            add_dict.update({"description": description.get(ch_types[idx])})
            add_list.append(add_dict)
        index = pd.Index(add_chs, name="name")
        df_add = pd.DataFrame(add_list, index=index)
        df_chs = df_chs.append(df_add, ignore_index=False)
    remov_chs = [ch for ch in old_chs if ch not in raw_copy.ch_names]
    if remov_chs:
        df_chs = df_chs.drop(remov_chs)
    df_chs = df_chs.reindex(raw_copy.ch_names)
    os.remove(channels_path.fpath)
    df_chs.to_csv(
        os.path.join(working_dir, channels_path.basename),
        sep="\t",
        na_rep="n/a",
        index=True,
    )

    # Rewrite **electrodes.tsv
    elec_files = []
    for file in os.listdir(working_dir):
        if file.endswith("_electrodes.tsv") and "_space-" in file:
            elec_files.append(os.path.join(working_dir, file))
    for elec_file in elec_files:
        df_chs = pd.read_csv(elec_file, sep="\t", index_col=0)
        old_chs = df_chs.index.tolist()
        add_chs = [ch for ch in raw.ch_names if ch not in old_chs]
        add_list = []
        for _ in add_chs:
            add_dict = {}
            add_dict.update({column: "n/a" for column in df_chs.columns})
            add_list.append(add_dict)
        index = pd.Index(add_chs, name="name")
        df_add = pd.DataFrame(add_list, index=index)
        df_chs = df_chs.append(df_add, ignore_index=False)
        os.remove(elec_file)
        df_chs.to_csv(
            os.path.join(elec_file), sep="\t", na_rep="n/a", index=True
        )
    # Remove temporary working folder
    shutil.rmtree(temp_root)
    # Check for success
    raw = mne_bids.read_raw_bids(bids_path, verbose=False)
    return raw


def _rewrite_events(
    in_path: mne_bids.BIDSPath, out_path: mne_bids.BIDSPath
) -> None:
    """Rewrite **events.tsv to new location.
    
    Arguments
    ---------
    curr_path : mne_bids.BIDSPath
        BIDSPath object for input path.
    out_path : mne_bids.BIDSPath
        BIDSPath object for output path.
    """
    original_path = in_path.copy().update(suffix="events").fpath
    out_path.copy().update(suffix="events")
    target_path = os.path.join(out_path.directory, out_path.basename + ".tsv")
    shutil.copyfile(original_path, target_path)


def bids_get_coords(
    fname: str, root: str, space: str = "MNI152NLin2009bAsym"
) -> pd.DataFrame:
    """Read *electrodes.tsv file and return as pandas DataFrame.
    
    Arguments
    ---------
    fname : str
        Path to data file for which the corresponding electrodes.tsv file should be read.
    root : str
        Root of the BIDS dataset
    space : str, default: "MNI152NLin2009bAsym"
    
    Returns
    -------
    pd.DataFrame

    """
    entities = mne_bids.get_entities_from_fname(os.path.basename(fname))
    entities["suffix"] = "electrodes"
    elec_path = mne_bids.BIDSPath(**entities, datatype="ieeg")
    elec_path.update(
        root=root,
        task=None,
        acquisition=None,
        run=None,
        extension=".tsv",
        space=space,
    )
    return pd.read_csv(elec_path.fpath, sep="\t", index_col=0)

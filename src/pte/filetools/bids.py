"""Module for handling datasets in BIDS-format."""

from collections import defaultdict
import shutil
from pathlib import Path
from typing import Optional, Union

import mne
import mne_bids
import mne_bids.copyfiles
import numpy as np
import pandas as pd
import pybv
from mne_bids.path import get_bids_path_from_fname

import pte.preprocessing.channels


def sub_med_stim_from_fname(
    fname: Path | str | mne_bids.BIDSPath,
) -> tuple[str, str, str]:
    entities = mne_bids.get_entities_from_fname(fname)
    sub = entities["subject"]
    if "On" in entities["session"]:
        med = "ON"
    elif "Off" in entities["session"]:
        med = "OFF"
    else:
        med = "n/a"
    if "On" in entities["acquisition"]:
        stim = "ON"
    elif "Off" in entities["acquisition"]:
        stim = "OFF"
    else:
        stim = "n/a"
    return sub, med, stim


def add_coord_column(
    df_chs: pd.DataFrame, ch_names: list[str], new_ch: str
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
    if new_ch == "auto_summation":
        new_ch = pte.preprocessing.channels.summation_channel_name(ch_names)
    elif new_ch == "auto_bipolar":
        new_ch = pte.preprocessing.channels.bipolar_channel_name(ch_names)
    coords = [
        df_chs.loc[ch_name][["x", "y", "z"]].values for ch_name in ch_names
    ]
    coords = np.array(coords, float).mean(axis=0).round(5)
    df_chs.loc[new_ch] = df_chs.loc[ch_names[0]]
    df_chs.loc[new_ch, "size"] = None
    df_chs.loc[new_ch, ["x", "y", "z"]] = coords
    df_chs.sort_index(axis=0, inplace=True)
    return df_chs


def overwrite_bids_file(
    raw: mne.io.Raw, bids_path: mne_bids.BIDSPath
) -> mne.io.BaseRaw:
    """Overwrite BIDS file in BrainVision format with given Raw instance.

    Parameters
    ----------
    raw : raw MNE object
        The raw MNE object for this function to write
    bids_path : BIDSPath MNE-BIDS object
        The MNE BIDSPath to the file to be overwritten
    Returns
    -------
    raw : raw MNE object or None
        The newly written raw object.
    """
    data = raw.get_data()
    sfreq = raw.info["sfreq"]
    ch_names = raw.ch_names
    tmpdir = bids_path.directory / "tmpdir"
    fname_base = "tmpfile"
    source_path = Path(tmpdir, fname_base + ".vhdr")

    # rewrite datafile
    pybv.write_brainvision(
        data=data,
        sfreq=sfreq,
        ch_names=ch_names,
        events=None,
        fname_base=fname_base,
        folder_out=tmpdir,
    )

    raw = mne.io.read_raw_brainvision(source_path)
    raw.info["line_freq"] = 50

    mapping_dict = _get_mapping_dict(raw.ch_names)
    raw.set_channel_types(mapping_dict)

    mne_bids.write_raw_bids(raw, bids_path, overwrite=True)

    # check for success
    raw = mne_bids.read_raw_bids(bids_path, verbose=False)

    shutil.rmtree(tmpdir)

    return raw


def _get_mapping_dict(ch_names: list[str]) -> dict:
    """Create dictionary for remapping channel types.

    Arguments
    ---------
    ch_names : list
        Channel names to be remapped.

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


def rewrite_bids_file(
    raw: mne.io.BaseRaw,
    bids_path: mne_bids.BIDSPath,
    reorder_channels: bool = True,
) -> mne.io.BaseRaw:
    """Overwrite BrainVision data in BIDS format that has been modified.

    Parameters
    ----------
    raw : mne.Raw object
        The raw MNE object for this function to write
    bids_path : mne_bids.BIDSPath
        The MNE BIDSPath to be overwritten
    reorder_channels : bool. Default: True
        Set to false if channels should not be reordered

    Returns
    -------
    raw_new : raw MNE object
        The newly written raw object.
    """
    current_path = bids_path.copy().update(
        suffix=bids_path.datatype, extension=".vhdr"
    )
    current_dir = current_path.directory

    # Create backup directory
    backup_dir = Path(current_dir, "backup")
    # Backup files
    _backup_files(current_path, backup_dir)

    try:
        # Create temporary working directory
        temp_dir = Path(backup_dir, "temp")
        temp_dir.mkdir(exist_ok=False)
        temp_path = current_path.copy().update(root=temp_dir)
        raw = raw.copy()
        raw.set_montage(None)

        if reorder_channels:
            raw = raw.reorder_channels(sorted(raw.ch_names))

        mne_bids.write_raw_bids(
            raw,
            bids_path=temp_path,
            allow_preload=True,
            format="BrainVision",
            verbose=False,
        )

        # Rewrite BrainVision files
        mne_bids.copyfiles.copyfile_brainvision(temp_path, current_path.fpath)

        # Rewrite *events.tsv
        _rewrite_events(temp_path, current_path)

        # Rewrite *channels.tsv
        _rewrite_channels(current_path, raw)

        # Rewrite *electrodes.tsv
        for file in Path(current_dir).glob("*electrodes.tsv"):
            _rewrite_electrodes(file=file, raw=raw)

    except Exception as exception:
        print("Rewriting failed, cleaning up...")
        for file in Path(backup_dir).glob("*"):
            if file.is_file():
                shutil.copy(file, current_dir)
        raise exception
    finally:
        # Clean up
        shutil.rmtree(backup_dir)

    # Check for success
    raw = mne_bids.read_raw_bids(bids_path, verbose=False)

    return raw


def _backup_files(current_path: mne_bids.BIDSPath, backup_dir: Path) -> None:
    """Create backup of BIDS files."""
    backup_dir.mkdir(exist_ok=True, parents=True)
    try:
        backup_path = (backup_dir / current_path.basename).with_suffix(".vhdr")
        mne_bids.copyfiles.copyfile_brainvision(
            current_path.fpath, backup_path
        )

        channels_path = current_path.copy()
        channels_path.update(suffix="channels", extension=".tsv")
        shutil.copy(channels_path.fpath, backup_dir)

        for file in Path(current_path.directory).glob("*electrodes.tsv"):
            shutil.copy(file, backup_dir)

        events_path = current_path.copy()
        events_path.update(suffix="events", extension=".tsv")
        if events_path.fpath.exists():
            shutil.copy(events_path.fpath, backup_dir)

    except Exception as error:
        shutil.rmtree(backup_dir)
        raise error


def _rewrite_events(
    in_path: mne_bids.BIDSPath, out_path: mne_bids.BIDSPath
) -> None:
    """Rewrite *events.tsv to new location.

    Arguments
    ---------
    in_path : mne_bids.BIDSPath
        Input path.
    out_path : mne_bids.BIDSPath
        Output path.
    """
    in_path = in_path.copy()
    in_path.update(suffix="events", extension=".tsv")
    out_path = out_path.copy()
    out_path.update(suffix="events", extension="tsv")
    shutil.copyfile(in_path.fpath, out_path.fpath)


def _get_description(channel_type: str) -> str:
    """Get channel type description."""
    description = defaultdict(lambda: "Other type of channel")
    description.update(
        meggradaxial="Axial Gradiometer",
        megrefgradaxial="Axial Gradiometer Reference",
        meggradplanar="Planar Gradiometer",
        megmag="Magnetometer",
        megrefmag="Magnetometer Reference",
        meg="MagnetoEncephaloGram",
        stim="Trigger",
        eeg="ElectroEncephaloGram",
        ecog="Electrocorticography",
        seeg="StereoEEG",
        ecg="ElectroCardioGram",
        eog="ElectroOculoGram",
        emg="ElectroMyoGram",
        misc="Miscellaneous",
        bio="Biological",
        ias="Internal Active Shielding",
        dbs="Deep Brain Stimulation",
    )
    return description[channel_type]


def _get_group(channel_name: str) -> str:
    """Get group for corresponding channel name."""
    sides = defaultdict(lambda: "")
    sides.update(L="left", R="right")
    groups = defaultdict(lambda: "n/a")
    groups.update(
        MEG="MEG",
        ACC="accelerometer",
        EEG="EEG",
        ECOG="ECOG",
        SEEG="SEEG",
        ECG="ECG",
        EOG="EOG",
        EMG="EMG",
        MISC="MISC",
        LFP="DBS",
    )
    items = channel_name.split("_")
    return f"{groups[items[0]]}_{sides[items[1]]}".strip("_")


def _rewrite_channels(
    bids_path: mne_bids.BIDSPath, raw: mne.io.BaseRaw
) -> None:
    """Update and rewrite electrodes.tsv file."""

    channels_path = bids_path.copy().update(
        suffix="channels", extension=".tsv"
    )

    data_old: pd.DataFrame = pd.read_csv(  # type: ignore
        channels_path.fpath, sep="\t", index_col=0
    )
    channels_old = data_old.index.tolist()
    channels_new = raw.ch_names

    channels_to_remove = [ch for ch in channels_old if ch not in raw.ch_names]

    channels_to_add = [ch for ch in channels_new if ch not in channels_old]

    if not any((channels_to_add, channels_to_remove)):
        return

    if channels_to_remove:
        data_old = data_old.drop(channels_to_remove)

    if channels_to_add:
        add_list = []
        ch_types = raw.get_channel_types(picks=channels_to_add)
        for ch_name, ch_type in zip(channels_to_add, ch_types):
            add_dict = {
                data_old.columns[i]: data_old.iloc[0][i]
                for i in range(len(data_old.columns))
            }
            add_dict.update(
                status="good",
                description=_get_description(ch_type),
                type=ch_type.upper(),
                group=_get_group(ch_name),
            )
            add_list.append(add_dict)
        print(f"Added channels: {channels_to_add}")
        print(f"Added channel types: {ch_types}")

        index = pd.Index(channels_to_add, name="name")
        data_to_add = pd.DataFrame(add_list, index=index)
        data_old = data_old.append(data_to_add, ignore_index=False)

    data_old = data_old.reindex(index=channels_new)
    data_old.to_csv(
        channels_path.fpath,
        sep="\t",
        na_rep="n/a",
        index=True,
        index_label="name",
    )


def _rewrite_electrodes(file: Union[str, Path], raw: mne.io.BaseRaw) -> None:
    """Update and rewrite electrodes.tsv file."""
    data_old: pd.DataFrame = pd.read_csv(  # type: ignore
        file, sep="\t", index_col=0
    )
    electrodes_old = data_old.index.tolist()

    electrodes_to_add = []
    for ch_name, ch_type in zip(raw.ch_names, raw.get_channel_types()):
        if (ch_name not in electrodes_old) and (
            ch_type in ["ecog", "dbs", "seeg", "eeg"]
        ):
            electrodes_to_add.append(ch_name)

    data_to_add = pd.DataFrame(
        data=None, index=electrodes_to_add, columns=data_old.columns
    )

    data_old = data_old.append(data_to_add, ignore_index=False)
    data_old.sort_index(axis=0, inplace=True)

    data_old.to_csv(
        file, sep="\t", na_rep="n/a", index=True, index_label="name"
    )


def get_bids_electrodes(
    fname: str, root: Optional[str] = None, space: str = "MNI152NLin2009bAsym"
) -> tuple[pd.DataFrame, mne_bids.BIDSPath]:
    """Read *electrodes.tsv file and return as pandas DataFrame.

    Arguments
    ---------
    fname : str
        Path to data file for which the corresponding electrodes.tsv file
        should be read.
    root : str
        Root of the BIDS dataset
    space : str, default: "MNI152NLin2009bAsym"

    Returns
    -------
    pd.DataFrame
    mne_bids.BIDSPath

    """
    elec_path = get_bids_path_from_fname(fname)
    elec_path.update(
        suffix="electrodes",
        extension=".tsv",
        space=space,
        run=None,
        task=None,
        acquisition=None,
    )
    if root:
        elec_path.update(root=root)
    return pd.read_csv(elec_path.fpath, sep="\t", index_col=0), elec_path

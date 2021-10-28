import os
import shutil

from numba import jit
import numpy as np
from plotly import express
import pandas as pd
import scipy.io as spio
from scipy.signal import decimate, detrend

import mne
import mne_bids
from pybv import write_brainvision


def add_coord_col(df, ch_names, new_ch):
    """"""
    coords = []
    for ch in ch_names:
        coords.append(df.loc[ch][['x', 'y', 'z']].values)
    coords = np.array(coords, float).mean(axis=0).round(5)
    append = df.loc[ch].copy()
    append.loc[['x', 'y', 'z']] = coords
    append.name = new_ch
    df = df.append(append, ignore_index=False)
    df.sort_index(axis=0, inplace=True)
    return df


def add_squared_channel(raw, event_id, ch_name):
    """Create squared data (0s and 1s) from events and add to Raw object.

    Parameters
    ----------
    raw : MNE Raw object
        The MNE Raw object for this function to modify.
    event_id : dict | callable() | None | ‘auto’
        event_id (see MNE documentation) defining the annotations to be chosen
        from your Raw object. ONLY pass annotation names that should be used to
        generate the squared data.
        Can be:
            dict: map descriptions (keys) to integer event codes (values). Only the descriptions present will be mapped, others will be ignored.
            callable: must take a string input and return an integer event code, or return None to ignore the event.
            None: Map descriptions to unique integer values based on their sorted order.
            ‘auto’ (default): prefer a raw-format-specific parser:
                Brainvision: map stimulus events to their integer part; response events to integer part + 1000; optic events to integer part + 2000; ‘SyncStatus/Sync On’ to 99998; ‘New Segment/’ to 99999; all others like None with an offset of 10000.
                Other raw formats: Behaves like None.
    ch_name : str
        Name for the squared channel to be added.

    Returns
    -------
    raw_final : MNE Raw object
        The Raw object containing the added squared channel. Is a copy of the
        original Raw object.
    """
    events, event_id = mne.events_from_annotations(
        raw, event_id)
    data = raw.get_data()
    evs_idx = events[:, 0]
    onoff = np.zeros((1, data.shape[1]))
    for i in np.arange(0, len(evs_idx), 2):
        onoff[0, evs_idx[i]:evs_idx[i + 1]] = 1
    info = mne.create_info(
        ch_names=[ch_name], ch_types=['misc'], sfreq=raw.info['sfreq'])
    raw_sq = mne.io.RawArray(onoff, info)
    raw_sq.info['meas_date'] = raw.info['meas_date']
    raw_sq.info['line_freq'] = 50
    raw_final = raw.copy().load_data().add_channels(
        [raw_sq], force_update_info=True)
    return raw_final


def add_summation_channel(raw, sum_channels, new_ch):
    """

    Parameters
    ----------
    raw
    sum_channels
    new_ch

    Returns
    -------
    raw_final : MNE Raw object
        The Raw object containing the added channel. Is a copy of the
        original Raw object.
    """
    data = raw.get_data(picks=sum_channels)
    new_data = np.expand_dims(data.sum(axis=0), axis=0)
    ch_type = raw.get_channel_types(picks=sum_channels[0])
    info = mne.create_info(
        [new_ch], sfreq=raw.info['sfreq'], ch_types=ch_type, verbose=False)
    raw_new = mne.io.RawArray(
        new_data, info, first_samp=0, copy='auto', verbose=False)
    raw_final = raw.copy().load_data().add_channels(
        [raw_new], force_update_info=True)
    return raw_final


def bids_save_file(raw, bids_path, return_raw=False):
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
    sfreq = raw.info['sfreq']
    ch_names = raw.ch_names
    folder = bids_path.directory
    # events, event_id = mne.events_from_annotations(raw)

    # rewrite datafile
    write_brainvision(
        data=data, sfreq=sfreq, ch_names=ch_names, events=None,
        fname_base='dummy', folder_out=folder)
    source_path = os.path.join(folder, 'dummy' + '.vhdr')
    raw = mne.io.read_raw_brainvision(source_path)
    raw.info['line_freq'] = 50
    remapping_dict = {}
    for ch_name in raw.ch_names:
        if ch_name.startswith('ECOG'):
            remapping_dict[ch_name] = 'ecog'
        elif ch_name.startswith(('LFP', 'STN')):
            remapping_dict[ch_name] = 'dbs'
        elif ch_name.startswith('EMG'):
            remapping_dict[ch_name] = 'emg'
        # mne_bids can't handle both eeg and ieeg channel types in the same data
        elif ch_name.startswith('EEG'):
            remapping_dict[ch_name] = 'eeg'
        elif ch_name.startswith(('MOV', 'ANALOG', 'ROT', 'ACC',
                                 'AUX', 'X', 'Y', 'Z', 'MISC')):
            remapping_dict[ch_name] = 'misc'
        else:
            remapping_dict[ch_name] = 'misc'
    raw.set_channel_types(remapping_dict)
    # mne_bids.write_raw_bids(raw, bids_path, events, event_id, overwrite=True)
    mne_bids.write_raw_bids(raw, bids_path, overwrite=True)
    suffixes = ['.eeg', '.vhdr', '.vmrk']
    dummy_files = [os.path.join(folder, 'dummy' + suffix)
                   for suffix in suffixes]
    for dummy_file in dummy_files:
        os.remove(dummy_file)
    # check for success
    raw = mne_bids.read_raw_bids(bids_path, verbose=False)
    if return_raw is True:
        return raw


def bids_rewrite_file(raw, bids_path, return_raw=False):
    """Overwrite BrainVision data in BIDS format that has been modified.

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
    raw_new : raw MNE object or None
        The newly written raw object.
    """
    curr_path = bids_path.copy().update(
        suffix=bids_path.datatype, extension=None)
    raw_copy = raw.copy()
    working_dir = curr_path.directory

    temp_root = os.path.join(working_dir, 'temp')
    if not os.path.isdir(temp_root):
        os.mkdir(temp_root)
    out_path = curr_path.copy().update(root=temp_root)
    raw_copy.info['dig'] = None
    mne_bids.write_raw_bids(raw_copy, out_path, allow_preload=True,
                            format='BrainVision', verbose=False)

    # Rewrite BrainVision files
    mne_bids.copyfiles.copyfile_brainvision(out_path, curr_path.fpath)

    # Rewrite **events.tsv
    original_path = out_path.copy().update(suffix='events').fpath
    events_path = curr_path.copy().update(suffix='events')
    target_path = os.path.join(
        events_path.directory, events_path.basename + '.tsv')
    shutil.copyfile(original_path, target_path)

    # Rewrite **channels.tsv
    channels_path = bids_path.copy().update(suffix='channels', extension='.tsv')
    df = pd.read_csv(channels_path.fpath, sep='\t', index_col=0)
    old_chs = df.index.tolist()
    add_chs = [ch for ch in raw.ch_names if ch not in old_chs]
    description = {'seeg': 'StereoEEG', 'ecog': 'Electrocorticography',
                   'eeg': 'Electroencephalography', 'emg': 'Electromyography',
                   'misc': 'Miscellaneous', 'dbs': 'Deep Brain Stimulation'}
    if add_chs:
        add_list = []
        ch_types = raw_copy.get_channel_types(picks=add_chs)
        print("Added channels: ", add_chs)
        print("Added channel types: ", ch_types)
        for idx, add_ch in enumerate(add_chs):
            add_dict = {}
            add_dict.update({df.columns[i]: df.iloc[0][i]
                             for i in range(0, len(df.columns))})
            add_dict.update({'type': ch_types[idx].upper()})
            add_dict.update({'description': description.get(ch_types[idx])})
            add_list.append(add_dict)
        index = pd.Index(add_chs, name='name')
        df_add = pd.DataFrame(add_list, index=index)
        df = df.append(df_add, ignore_index=False)
    remov_chs = [ch for ch in old_chs if ch not in raw_copy.ch_names]
    if remov_chs:
        df = df.drop(remov_chs)
    df = df.reindex(raw_copy.ch_names)
    os.remove(channels_path.fpath)
    df.to_csv(os.path.join(working_dir, channels_path.basename),
              sep='\t', na_rep='n/a', index=True)

    # Rewrite **electrodes.tsv
    elec_files = []
    for file in os.listdir(working_dir):
        if file.endswith('_electrodes.tsv') and '_space-' in file:
            elec_files.append(os.path.join(working_dir, file))
    for elec_file in elec_files:
        df = pd.read_csv(elec_file, sep='\t', index_col=0)
        old_chs = df.index.tolist()
        add_chs = [ch for ch in raw.ch_names if ch not in old_chs]
        add_list = []
        for _ in add_chs:
            add_dict = {}
            add_dict.update({column: 'n/a' for column in df.columns})
            add_list.append(add_dict)
        index = pd.Index(add_chs, name='name')
        df_add = pd.DataFrame(add_list, index=index)
        df = df.append(df_add, ignore_index=False)
        os.remove(elec_file)
        df.to_csv(os.path.join(elec_file), sep='\t', na_rep='n/a', index=True)
    # Remove temporary working folder
    shutil.rmtree(temp_root)
    # Check for success
    raw = mne_bids.read_raw_bids(bids_path, verbose=False)
    if return_raw:
        return raw


def get_all_files(path, suffix, get_bids=False, prefix=None, bids_root=None,
                  verbose=False, extension=None):
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
                "Warning: No root folder given. Please pass bids_root parameter to create a complete BIDS_Path object.")
        bids_paths = []
        for filepath in filepaths:
            entities = mne_bids.get_entities_from_fname(filepath)
            try:
                bids_path = mne_bids.BIDSPath(subject=entities["subject"],
                                              session=entities["session"],
                                              task=entities["task"],
                                              run=entities["run"],
                                              acquisition=entities[
                                                  "acquisition"],
                                              suffix=entities["suffix"],
                                              extension=extension,
                                              root=bids_root)
            except ValueError as err:
                print(
                    f"ValueError while creating BIDS_Path object for file {filepath}: {err}")
            else:
                bids_paths.append(bids_path)

    if verbose:
        if not bids_paths:
            print("No corresponding files found.")
        else:
            print('Corresponding files found:')
            for idx, file in enumerate(bids_paths):
                print(idx, ':', os.path.basename(file))

    return bids_paths


def get_coords_bids(fname, root, space='MNI152NLin2009bAsym'):
    """"""
    entities = mne_bids.get_entities_from_fname(os.path.basename(fname))
    entities['suffix'] = 'electrodes'
    elec_path = mne_bids.BIDSPath(**entities, datatype='ieeg')
    elec_path.update(
        root=root, task=None, acquisition=None, run=None,
        extension='.tsv', space=space)
    return pd.read_csv(elec_path.fpath, sep='\t', index_col=0)


def raw_plotly(mne_raw, file_name, t_slice=(), plot_title=None,
               do_decimate=True, do_normalize=True, do_detrend="linear", padding=2):
    """
    Creates (exports) the (sliced) MNE raw signal as an HTML plotly plot

    Arguments:
        mne_raw: MNE raw object (output of mne.io.read_raw_...)
        file_name: name (and directory) for the exported html file
        t_slice: tuple of `start` and `end` slice (seconds)
            example: `t_slice = (1, 5)` returns the 1s-5s slice
        plot_title: Plot title (default is None)
        do_decimate: down-sampling (decimating) the signal to 200Hz sampling rate
            (default and recommended value is True)
        do_normalize: dividing the signal by the root mean square value for normalization
            (default and recommended value is True)
        do_detrend: The type of detrending.
            If do_detrend == 'linear' (default), the result of a linear least-squares fit to data is subtracted from data.
            If do_detrend == 'constant', only the mean of data is subtracted.
            else, no detrending
        padding: multiplication factor for spacing between signals on the y-axis
            For highly variant data, use higher values. default is 2

    returns nothing
    """
    samp_freq = int(mne_raw.info["sfreq"])
    channels_array = np.array(mne_raw.info["ch_names"])
    if t_slice:
        signals_array, time_array = mne_raw[:, t_slice[0]*samp_freq:t_slice[1]*samp_freq]
    else:
        signals_array, time_array = mne_raw[:, :]

    sig_plotly(time_array, signals_array, channels_array, samp_freq, file_name, plot_title=plot_title,
               do_decimate=do_decimate, do_normalize=do_normalize, do_detrend=do_detrend, padding=padding)


def rms(data, axis=-1):
    """
    returns the Root Mean Square (RMS) value of data along the given axis
    """
    assert axis < data.ndim, \
        "No {} axis for data with {} dimension!".format(axis, data.ndim)
    if axis < 0:
        return np.sqrt(np.mean(np.square(data)))
    else:
        return np.sqrt(np.mean(np.square(data), axis=axis))


def sig_plotly(time_array, signals_array, channels_array, samp_freq, file_name,
               plot_title=None,
               do_decimate=True, do_normalize=True, do_detrend="linear",
               padding=2):
    """
    Creates (exports) the signals as an HTML plotly plot

    Arguments:
        time_array: numpy array of time stamps (seconds)
        signals_array: a 2D-array of signals with shape (#channels, #samples)
        channels_array: numpy array (or list) of channel names
        samp_freq: sampling frequency (Hz)
        file_name: name (and directory) for the exported html file
        plot_title: Plot title (default is None)
        do_decimate: down-sampling (decimating) the signal to 200Hz sampling rate
            (default and recommended value is True)
        do_normalize: dividing the signal by the root mean square value for normalization
            (default and recommended value is True)
        do_detrend: The type of detrending.
            If do_detrend == 'linear' (default), the result of a linear least-squares fit to data is subtracted from data.
            If do_detrend == 'constant', only the mean of data is subtracted.
            else, no detrending
        padding: multiplication factor for spacing between signals on the
            y-axis. For highly variant data, use higher values. default is 2.

    returns nothing
    """

    time_array = np.squeeze(time_array)
    signals_array = np.squeeze(signals_array)
    channels_array = np.squeeze(channels_array)
    if signals_array.ndim == 1:
        signals_array = signals_array.reshape(1, -1)

    assert signals_array.shape[0] == channels_array.shape[0], \
        "signals_array ! channels_array Dimension mismatch!"
    assert signals_array.shape[1] == time_array.shape[0], \
        "signals_array ! time_array Dimension mismatch!"

    if do_decimate:
        decimate_factor = min(10, int(samp_freq / 200))
        signals_array = decimate(signals_array, decimate_factor)
        time_array = decimate(time_array, decimate_factor)
    if do_detrend == "linear" or do_detrend == "constant":
        signals_array = detrend(signals_array, axis=1, type=do_detrend,
                                overwrite_data=True)
    if do_normalize:
        eps_ = np.finfo(float).eps
        signals_array = signals_array / (
                    rms(signals_array, axis=1).reshape(-1, 1) + eps_)

    offset_value = padding * rms(signals_array)  # RMS value
    signals_array = signals_array + offset_value * (
        np.arange(len(channels_array)).reshape(-1, 1))

    signals_df = pd.DataFrame(data=signals_array.T, index=time_array,
                              columns=channels_array)

    fig = express.line(signals_df, x=signals_df.index, y=signals_df.columns,
                       line_shape="spline", render_mode="svg",
                       labels=dict(index="Time (s)",
                                   value="(a.u.)",
                                   variable="Channel"), title=plot_title)
    fig.update_layout(yaxis=dict(tickmode='array',
                                 tickvals=offset_value * np.arange(
                                     len(channels_array)),
                                 ticktext=channels_array))

    fig.write_html(str(file_name) + ".html")


@jit(nopython=True)
def threshold_events(data, thresh):
    """Apply threshold to find start and end of events.
    """

    onoff = np.where(data > thresh, 1, 0)
    onoff_diff = np.zeros_like(onoff)
    onoff_diff[1:] = np.diff(onoff)
    index_start = np.where(onoff_diff == 1)[0]
    index_stop = np.where(onoff_diff == -1)[0]
    arr_start = np.stack((index_start, np.zeros_like(index_start),
                          np.ones_like(index_start)), axis=1)
    arr_stop = np.stack((index_stop, np.zeros_like(index_stop),
                         np.ones_like(index_stop) * -1), axis=1)
    return np.vstack((arr_start, arr_stop))


def loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)


def _check_keys(dict):
    '''
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    '''
    for key in dict:
        if isinstance(dict[key], spio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict


def _todict(matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries
    '''
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, spio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict

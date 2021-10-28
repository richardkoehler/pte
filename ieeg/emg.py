from numba import jit
import numpy as np

import mne


def get_emg_rms(
        raw, emg_ch, window_len, analog_ch, scaling=1e0, rereference=False,
        notch_filter=50):
    """Return root mean square with given window length of raw object. 
    
     Parameters
    ----------
    raw : MNE raw object
        The data to be processed.
    emg_ch : list of str
        The EMG channels to be processed. Must be of length 1 or 2.
    window_len : float | int | array-like of float/int
        Window length(s) for root mean square calculation in milliseconds.
    analog_ch : str | list of str
        The target channel (e.g., rotameter) to be added to output raw object.
    rereference : boolean (optional)
        Set to True if EMG channels should be referenced in a bipolar montage.
        Default is False.

    Returns
    -------
    raw_rms : MNE raw object
        Raw object containing root mean square of windowed signal and target
        channel.
    """
    
    raw_emg = raw.copy().pick(picks=emg_ch).load_data()
    raw_emg.set_channel_types(
        mapping={name: 'eeg' for name in raw_emg.ch_names})
    if rereference:
        raw_emg = mne.set_bipolar_reference(
            raw_emg, anode=raw_emg.ch_names[0], cathode=raw_emg.ch_names[1],
            ch_name=['EMG_BIP'], drop_refs=True, copy=False)
    data_bip = raw_emg.get_data()[0]
    if notch_filter:
        assert isinstance(notch_filter, (int, float))
        freqs = np.arange(notch_filter, 500, notch_filter)
        raw_emg = raw_emg.notch_filter(freqs, verbose=None)
    raw_filt = raw_emg.filter(l_freq=15, h_freq=500, verbose=False)
    if isinstance(window_len, (int, float)):
        window_len = [window_len]
    data = raw_filt.get_data()[0]
    data_arr = np.empty((len(window_len), len(data)))
    for idx, window in enumerate(window_len):
        data_rms = rms_window_nb(data, window, raw.info['sfreq'])
        data_rms_zx = (data_rms-np.mean(data_rms))/np.std(data_rms)
        data_arr[idx, :] = data_rms_zx * scaling
    data_analog = raw.copy().pick(picks=analog_ch).get_data()[0]
    if np.abs(min(data_analog)) > max(data_analog):
        data_analog = data_analog*-1
    data_all = np.vstack((data_analog, data_bip, data_arr))
    emg_ch_names = ['EMG_RMS_' + str(window) for window in window_len]
    info_rms = mne.create_info(
        ch_names=[analog_ch] + ['EMG_BIP'] + emg_ch_names, ch_types='emg',
        sfreq=raw.info['sfreq'])
    raw_rms = mne.io.RawArray(data_all, info_rms)
    raw_rms.info['meas_date'] = raw.info['meas_date']
    raw_rms.info['line_freq'] = raw.info['line_freq']
    raw_rms.set_annotations(raw.annotations)
    raw_rms.set_channel_types({analog_ch: 'misc'})
    raw_rms._orig_units = {ch: 'µV' for ch in raw_rms.ch_names}
    return raw_rms
    

@jit(nopython=True)
def rms_window_nb(data, window_len, sfreq):
    """Return root mean square of input signal with given window length.
    
     Parameters
    ----------
    data : array
        The data to be processed. Must be 1-dimensional.
    window_len : float | int
        Window length in milliseconds.
    sfreq : float | int
        Sampling frequency in 1/seconds.

    Returns
    -------
    data_rms
        Root mean square of windowed signal. Same dimension as input signal
    """
    
    half_window_size = int(sfreq * window_len / 1000 / 2)
    data_rms = np.empty_like(data)
    for i in range(len(data)):
        if i == 0 or i == len(data)-1:
            data_rms[i] = np.absolute(data[i])
        elif i < half_window_size:
            new_window_size = i
            data_rms[i] = np.sqrt(np.mean(np.power(
                data[i-new_window_size:i+new_window_size], 2)))
        elif len(data)-i < half_window_size:
            new_window_size = len(data)-i
            data_rms[i] = np.sqrt(np.mean(np.power(
                data[i-new_window_size:i+new_window_size], 2)))
        else:
            data_rms[i] = np.sqrt(np.mean(np.power(
                data[i-half_window_size:i+half_window_size], 2)))
    return data_rms

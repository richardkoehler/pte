"""Module to read .mat files."""

import scipy


def loadmat(filename: str) -> dict:
    """Read .mat file and convert to dictionary.

    This function should be called instead of direct scipy.io.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects.

    Arguments
    ---------
    filename : str
        Complete filepath of .mat file.

    Returns
    -------
    dict
        Data read from .mat file.
    """
    data = scipy.io.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)


def _check_keys(data: dict) -> dict:
    """
    Check if entries in dictionary are mat-objects. If yes
    call _todict to change them to nested dictionaries.
    """
    for key in data:
        if isinstance(data[key], scipy.io.matlab.mio5_params.mat_struct):
            data[key] = _todict(data[key])
    return data


def _todict(matobj: scipy.io.matlab.mio5_params.mat_struct) -> dict:
    """
    Construct nested dictionary from matobject.

    Arguments
    ---------
    matobj : scipy.io.matlab.mio5_params.mat_struct
        mat_struct to be converted to nested dictionary.

    Returns
    -------
    dict
        mat_struct converted to nested dictionary.
    """
    data = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, scipy.io.matlab.mio5_params.mat_struct):
            data[strg] = _todict(elem)
        else:
            data[strg] = elem
    return dict

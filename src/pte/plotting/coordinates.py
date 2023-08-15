"""Module for handling brain coordinates."""
import pathlib

import numpy as np
import pandas as pd
import scipy

RESOURCES = pathlib.Path(__file__).parent / "resources"


#  Original matlab code from Xu Cui (https://alivelearn.net/?p=1456)
#  This python code is adapted from Astrid Olave
#  (https://github.com/aaolaveh/anat-from-MNI/blob/master/functions.py)
def find_structure_mni(
    mni_coords: np.ndarray, database: pathlib.Path | str | None = None
) -> list[str] | str:
    """
    Convert MNI coordinate to a description of brain structure in aal

    Arguments
    ---------
    mni_coords : numpy array
        The MNI coordinates of given points, in mm. An Mx3 matrix
        (shape (M, 3)) or vector of shape (3,) where each row is the
        coordinates (x, y, z) of one point.
    database : Path or str, optional
        Path to database. If none given, `TDdatabase.mat` is used.

    Returns
    -------
    descriptions:
        A list of M elements, each describing each point.
    """
    if mni_coords.ndim == 1:
        mni_coords = np.expand_dims(mni_coords, axis=0)

    if not isinstance(mni_coords, np.ndarray):
        mni_coords = np.ndarray(mni_coords)

    if mni_coords.size == 0 | mni_coords.shape[-1] != 3:
        raise ValueError(
            "The given coordinates are not 3-length coordinates. The last "
            "dimension must be of size 3. Got `mni_coords` shape:"
            f" {mni_coords.shape}"
        )

    if database is None:
        atlas = scipy.io.loadmat(str(RESOURCES / "TDdatabase.mat"))

    ind_coords = mni2coor(
        mni_coords=mni_coords,
    )
    # -1 by python indexation
    ind_coords = ind_coords - 1

    rows = np.shape(atlas["DB"])[1]
    descriptions = []
    for ind in ind_coords:
        single_result = []
        for j in range(rows):
            # atlas["DB"][0,j][0,0][0] is the j-th 3D-matrix
            graylevel = atlas["DB"][0, j][0, 0][0][ind[0], ind[1], ind[2]]
            if graylevel == 0:
                label = "undefined"
            else:
                if j < (rows - 1):
                    suffix = ""
                else:
                    suffix = " (aal)"

                # mat['DB'][0,j][0,0][1]  is the list with regions
                label = (
                    atlas["DB"][0, j][0, 0][1][0, (graylevel - 1)][0] + suffix
                )

            single_result.append(label)
        descriptions.append(single_result)
    return descriptions


def mni2coor(
    mni_coords: np.ndarray, matrix: np.ndarray | None = None
) -> np.ndarray:
    """
    Convert mni coordinates to matrix coordinates.

    Arguments
    ---------
    mni_coords : numpy array
        The MNI coordinates of given points, in mm. An Mx3 matrix
        where each row is the coordinates (x, y, z) for one point.

    matrix : numpy array, optional
        Transformation matrix. If None given, defaults to:
            [[2,   0,   0,  -92],
             [0,   2,   0, -128],
             [0,   0,   2,  -74],
             [0,   0,   0,    1]]

    Returns
    -------
    coords : numpy array
        Coordinate matrix
    """
    if matrix is None:
        matrix = np.array(
            [[2, 0, 0, -92], [0, 2, 0, -128], [0, 0, 2, -74], [0, 0, 0, 1]]
        )
        # This matrix is a remnant of the original code - function unclear
        # matrix = np.array(
        #     [[-4, 0, 0, 84], [0, 4, 0, -116], [0, 0, 4, -56], [0, 0, 0, 1]]
        # )

    ones = np.ones((np.shape(mni_coords)[0], 1))
    vector = np.hstack((mni_coords, ones))

    matrix_transp = np.transpose(np.linalg.inv(matrix))
    coords = vector.dot(matrix_transp)[:, 0:3]

    vround = np.vectorize(matlab_round)
    return vround(coords)


def matlab_round(value: int | float) -> int:
    """Round value to integer like round function in MATLAB.

    Arguments
    ---------
    data: float
        value to be rounded

    Returns
    -------
    Rounded value
    """
    if value - np.floor(value) != 0.5:
        return round(value)
    if value < 0:
        return int(value - 0.5)
    return int(value + 0.5)


def add_coords(data: pd.DataFrame, coords: pd.DataFrame) -> pd.DataFrame:
    """Add x, y and z electrode coordinates to DataFrame."""
    data.loc[:, ["x", "y", "z"]] = None
    for ind in data.index:
        if "avgref" in ind[1]:
            ind_elec = (ind[0], ind[1][:-7])
        else:
            ind_elec = ind
        try:
            data.loc[ind, ["x", "y", "z"]] = coords.loc[
                ind_elec, ["x", "y", "z"]
            ]
        except KeyError as error:
            print(
                f"KeyError raised by pandas. The following Subject and "
                f"Channel Name combination was not found: {error}."
            )
    return data

"""Module for handling brain coordinates."""
import pandas as pd


def add_coords(data: pd.DataFrame, coords: pd.DataFrame) -> pd.DataFrame:
    """Add x, y and z electrode coordinates to DataFrame."""
    data.loc[:, ["x", "y", "z"]] = None
    for ind in data.index:
        if "avgref" in ind[1]:
            ind_elec = (ind[0], ind[1][:-7])
        else:
            ind_elec = ind
        try:
            test = data.loc[ind, ["x", "y", "z"]]
            coor = coords.loc[ind_elec, ["x", "y", "z"]]
            data.loc[ind, ["x", "y", "z"]] = coords.loc[
                ind_elec, ["x", "y", "z"]
            ]
        except KeyError as e:
            print(
                f"KeyError raised by pandas. The following Subject and "
                f"Channel Name combination was not found: {e}."
            )
    return data

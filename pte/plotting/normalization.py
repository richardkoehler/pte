"""Module for scaling and normalizing data before plotting."""

from typing import Iterable, Optional, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def scale_minmax(
    data: pd.DataFrame,
    key_data: str,
    keys_index: Optional[Union[str, Iterable]] = None,
) -> pd.DataFrame:
    """Group data by given index keys and scale data with MinMaxScaler."""
    for ind, row in data.iterrows():
        data.loc[ind, "Channel Type"] = (
            "ECOG" if "ECOG" in row["Channel Name"] else "LFP"
        )
    if keys_index:
        if isinstance(keys_index, str):
            keys_index = [keys_index]
    data = data.set_index(keys=keys_index)
    data = data.sort_index()
    for ind in data.index:
        min_max_scaler = MinMaxScaler(feature_range=(0, 1))
        data_ = data.loc[ind, key_data].to_numpy(dtype=np.float32)
        if data_.size == 1:
            data_ = data_.reshape(1, -1)
        else:
            data_ = data_.reshape(-1, 1)
        data_scaled = min_max_scaler.fit_transform(X=data_).squeeze().round(2)
        data.loc[ind, key_data] = data_scaled
    return data

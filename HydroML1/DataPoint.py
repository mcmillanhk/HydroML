import pandas as pd
import numpy as np
from typing import List
import torch


class DataPoint:
    def __init__(self, gauge_id, flow_data, flow_data_cols, climate_data, climate_data_cols, signatures, attributes, latlong):
        self.gauge_id: List[str] = gauge_id

        #Maybe flow/climate should stay as dataframes? Actually tensors make more sense for all input...
        self.flow_data: np.array = flow_data
        self.flow_data_cols: List[str] = flow_data_cols
        self.climate_data: np.array = climate_data
        self.climate_data_cols: List[str] = climate_data_cols
        self.signatures: pd.DataFrame = signatures
        self.attributes: pd.DataFrame = attributes
        self.latlong: pd.DataFrame = latlong

    def hydro_data(self):
        return np.concatenate(self.flow_data, self.climate_data)

    def signatures_tensor(self):
        return torch.tensor(np.array(self.signatures))


def collate_fn(datapoints: List[DataPoint]):
    keys = [d.gauge_id for d in datapoints]
    return DataPoint(keys,
                     np.dstack([d.flow_data for d in datapoints]),
                     datapoints[0].flow_data_cols,
                     np.dstack([d.climate_data for d in datapoints]),
                     datapoints[0].climate_data_cols,
                     pd.concat([d.signatures for d in datapoints], keys=keys),
                     pd.concat([d.attributes for d in datapoints], keys=keys),
                     pd.concat([d.latlong for d in datapoints], keys=keys))


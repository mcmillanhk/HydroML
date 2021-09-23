import pandas as pd
import numpy as np
from typing import List
import torch


class DataPoint:
    def __init__(self, gauge_id, flow_data, flow_data_cols, climate_data, climate_data_cols, signatures, attributes, latlong):
        self.gauge_id: List[str] = gauge_id

        #Maybe flow/climate should stay as dataframes? Actually tensors make more sense for all input...
        self.flow_data: torch.tensor = flow_data
        self.flow_data_cols: List[str] = flow_data_cols
        self.climate_data: torch.tensor = climate_data
        self.climate_data_cols: List[str] = climate_data_cols
        self.signatures: pd.DataFrame = signatures
        self.attributes: pd.DataFrame = attributes
        self.latlong: pd.DataFrame = latlong

    def batch_size(self):
        return self.flow_data.shape[0]

    def timesteps(self):
        return self.flow_data.shape[1]

    def hydro_data(self):
        return torch.stack(self.flow_data, self.climate_data) # dim?

    def signatures_tensor(self):
        return torch.tensor(np.array(self.signatures))


def collate_fn(datapoints: List[DataPoint]):
    keys = [d.gauge_id for d in datapoints]
    return DataPoint(keys,
                     torch.stack([d.flow_data for d in datapoints], dim=0), #np.dstack([d.flow_data for d in datapoints]),
                     datapoints[0].flow_data_cols,
                     torch.stack([d.climate_data for d in datapoints], dim=0), #np.dstack(...),
                     datapoints[0].climate_data_cols,
                     pd.concat([d.signatures for d in datapoints], keys=keys),
                     pd.concat([d.attributes for d in datapoints], keys=keys),
                     pd.concat([d.latlong for d in datapoints], keys=keys))


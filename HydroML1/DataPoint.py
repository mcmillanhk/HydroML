import pandas as pd
import numpy as np
from typing import List
import torch


class DataPoint:
    def __init__(self, gauge_id, hydro_data, hydro_data_cols, signatures, attributes):
        self.gauge_id: List[str] = gauge_id
        #self.hydro_data: pd.DataFrame = hydro_data
        self.hydro_data: np.array = hydro_data
        self.hydro_data_cols: List[str] = hydro_data_cols
        self.signatures: pd.DataFrame = signatures
        self.attributes: pd.DataFrame = attributes

    def signatures_tensor(self):
        return torch.tensor(np.array(self.signatures))

    #def __init__(self, datapoints):
    #    self.gauge_id: str = [d.gauge_id for d in datapoints]
    #    self.hydro_data: pd.DataFrame = pd.concat([d.hydro_data for d in datapoints], axis=2)
    #    self.signatures: pd.DataFrame = pd.concat([d.signatures for d in datapoints], axis=2)
    #    self.attributes: pd.DataFrame = pd.concat([d.attributes for d in datapoints], axis=2)


def collate_fn(datapoints: List[DataPoint]):
    keys = [d.gauge_id for d in datapoints]
    return DataPoint(keys,
                     np.dstack([d.hydro_data for d in datapoints]),
                     datapoints[0].hydro_data_cols,
                     pd.concat([d.signatures for d in datapoints], keys=keys),
                     pd.concat([d.attributes for d in datapoints], keys=keys))


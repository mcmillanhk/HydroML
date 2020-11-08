from enum import Enum
from DataPoint import *
import pandas as pd
import numpy as np
import torch
from typing import List


class EncType(Enum):
    NoEncoder = 0
    LSTMEncoder = 1
    CNNEncoder = 2  # not implemented yet


class DatasetProperties:
    years_per_sample = 2
    length_days: int = 365 * years_per_sample

    #  attribs: dict of attrib/normalizing factor for each site
    attrib_normalizers = {'carbonate_rocks_frac': 1.0,  # Numbers, not categories, from geol.
               'geol_porostiy': 1.0,
               'geol_permeability': 0.1,
               'soil_depth_pelletier': 0.1,  # everything from soil
               'soil_depth_statsgo': 1,
               'soil_porosity': 1,
               'soil_conductivity': 1,
               'max_water_content': 1,
               'sand_frac': 0.01,  # These are percentages
               'silt_frac': 0.01,
               'clay_frac': 0.01,
               'water_frac': 0.01,
               'organic_frac': 0.01,
               'other_frac': 0.01,
               'elev_mean': 0.001,  # topo: not lat/long
               'slope_mean': 0.01,
               'area_gages2': 0.001,  # only the reliable of the 2 areas
               'gvf_max': 1,  # leaf area index seems totally correlated with these
               'gvf_diff': 1,
               }
    sig_normalizers = {
        # 'gauge_id': 1,
        'q_mean': 1,
        'runoff_ratio': 1,
        # 'slope_fdc': 1,
        'baseflow_index': 1,
        'stream_elas': 1,
        'q5': 1,
        'q95': 1,
        'high_q_freq': 0.05,
        'high_q_dur': 0.05,
        'low_q_freq': 0.05,
        'low_q_dur': 0.01,
        'zero_q_freq': 0.05,
        'hfd_mean': 0.01,
    }
    def num_sigs(self): return len(self.sig_normalizers)

    def sig_index(self, name): return list(self.sig_normalizers.keys()).index(name)


class EncoderProperties:
    encoder_type = EncType.CNNEncoder
    encoder_names = ["prcp(mm/day)", 'flow(cfs)', "tmax(C)"]  # "swe(mm)",
    # encoder_indices = get_indices(encoder_names, hyd_data_labels)
    # indices = list(hyd_data_labels).index()
    def encoder_input_dim(self): return len(self.encoder_names)

    encoding_num_layers = 2
    encoding_hidden_dim = 100

    def encoding_dim(self):
        return 0 if self.encoder_type == EncType.NoEncoder else 25

    def select_one_encoder_inputs(self, datapoint: DataPoint):
        datapoint.hydro_data: pd.DataFrame
        return np.array(datapoint.hydro_data(self.encoder_names))

    def encoder_perm(self):
        if self.encoder_type == EncType.LSTMEncoder:
            return (0, 2, 1)  # t x i x b -> t x b x i
        elif self.encoder_type == EncType.CNNEncoder:
            return (2, 1, 0)  # t x i x b -> b x i x t
        else:
            raise Exception("Encoder disabled or case not handled")

    def select_encoder_inputs(self, datapoint: DataPoint):

        return torch.tensor(datapoint.hydro_data[:, [datapoint.hydro_data_cols.index(name)
                                                     for name in self.encoder_names], :]).permute(self.encoder_perm())
        #return [self.select_one_encoder_inputs(datapoint) for datapoint in datapoints]

class DecoderType(Enum):
    LSTM = 0
    ConvNet = 1
    HydModel = 2

# Properties common to all encoders.
class DecoderProperties:

    decoder_model_type = DecoderType.HydModel

    #Properties specific to HydModelNet
    class HydModelNetProperties:
        class Indices:  # TODO still hardcoded in expected_b code
            STORE_DIM = 4  # 4 is probably the minimum: snow, deep, shallow, runoff
            SLOW_STORE = 0
            SNOW_STORE = STORE_DIM-1

    hyd_model_net_props = HydModelNetProperties()


def get_indices(encoder_names, hyd_data_labels):
    indices = [i for i, x in enumerate(hyd_data_labels) if x in encoder_names]
    #indices = []
    #for i, x in enumerate(hyd_data_labels):
    #    if x in encoder_names:
    #        indices = [indices, i]
    if len(indices) != len(encoder_names):
        raise Exception()
    return indices


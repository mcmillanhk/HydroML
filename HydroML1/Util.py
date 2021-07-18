from enum import Enum
from DataPoint import *
import pandas as pd
import numpy as np
import torch
from typing import List


class EncType(Enum):
    NoEncoder = 0
    LSTMEncoder = 1
    CNNEncoder = 2


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

    climate_norm = {
        'dayl(s)': 0.00002,
        'prcp(mm/day)': 1,  # we assume this is 1 in a couple places
        'srad(W/m2)': 0.005,
        'swe(mm)': 1,
        'tmax(C)': 0.1,
        'tmin(C)': 0.1,
        'vp(Pa)': 0.001,
    }

    def num_sigs(self): return len(self.sig_normalizers)

    def sig_index(self, name): return list(self.sig_normalizers.keys()).index(name)

    def temperatures(self, datapoint: DataPoint):
        temps = np.zeros([datapoint.climate_data.shape[0], 2, datapoint.climate_data.shape[2]])
        idx = get_indices(['tmin(C)'], datapoint.climate_data_cols)[0]
        temps[:, 0, :] = datapoint.climate_data[:, idx, :] / self.climate_norm['tmin(C)']
        idx = get_indices(['tmax(C)'], datapoint.climate_data_cols)[0]
        temps[:, 1, :] = datapoint.climate_data[:, idx, :] / self.climate_norm['tmax(C)']
        return temps

    def runoff_ratio(self, datapoint: DataPoint):
        rr=self.get_sig(datapoint, 'runoff_ratio')
        if rr.min() <= 0 or rr.max() >= 1:
            raise Exception(f"Runoff ratio outside reasonable range min={rr.min()} max={rr.max()}")
        return rr

    def get_sig(self, datapoint: DataPoint, sig: str):
        #rr = np.zeros([datapoint.signatures.shape[0]])  #  # batches
        idx = get_indices([sig], datapoint.signatures)[0]
        return datapoint.signatures[sig].to_numpy() / self.sig_normalizers[sig]


    def get_rain(self, datapoints: DataPoint):
        idx_rain = get_indices(['prcp(mm/day)'], self.climate_norm.keys())[0]
        return torch.from_numpy(datapoints.climate_data[:, idx_rain, :])  # rain is t x b


class EncoderProperties:
    encoder_type = EncType.CNNEncoder
    encoder_names = ["prcp(mm/day)", "tmax(C)"]  # "swe(mm)",'flow(cfs)',
    # encoder_indices = get_indices(encoder_names, hyd_data_labels)
    # indices = list(hyd_data_labels).index()
    def encoder_input_dim(self): return len(self.encoder_names)+1  # +1 for flow

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

    def select_encoder_inputs(self, datapoint: DataPoint, dataset_properties: DatasetProperties):
        indices = get_indices(self.encoder_names, dataset_properties.climate_norm.keys())
        return torch.tensor(np.concatenate((datapoint.flow_data, datapoint.climate_data[:, indices, :]), axis=1))
        #return torch.tensor(datapoint.hydro_data[:, [datapoint.flow_data_cols.index(name)
        #                                             for name in self.encoder_names], :]).permute(self.encoder_perm())
        #return [self.select_one_encoder_inputs(datapoint) for datapoint in datapoints]

class DecoderType(Enum):
    LSTM = 0
    ConvNet = 1
    HydModel = 2

# Properties common to all decoders.
class DecoderProperties:

    decoder_model_type = DecoderType.HydModel

    #Properties specific to HydModelNet
    class HydModelNetProperties:
        class Indices:
            STORE_DIM = 4  # 4 is probably the minimum: snow, deep, shallow, runoff
            SLOW_STORE = 0
            SLOW_STORE2 = 1 # a bit faster
            SURFACE_STORE = 2
            SNOW_STORE = STORE_DIM-1

        def __init__(self):
            #store_weights normalize stores when used as input
            self.store_weights = torch.ones((1, self.Indices.STORE_DIM))*0.001
            self.store_weights[0, self.Indices.SURFACE_STORE]=0.1
            self.store_weights[0, self.Indices.SNOW_STORE]=0.01

            # If scale_b then outflow_weights weight output from b so that we're predicting numbers with similar
            # magnitude for all pathways (first STORE_DIM^2 are between stores (if flow_between_stores), then STORE_DIM
            # are loss, then the final STORE_DIM are the store outputs
            self.outflow_weights = torch.ones((1, self.b_length()))*0.01

            store_outflow_weights = torch.ones((self.Indices.STORE_DIM))
            store_outflow_weights[self.Indices.SLOW_STORE]=0.01
            store_outflow_weights[self.Indices.SLOW_STORE2] = 0.05
            store_outflow_weights[self.Indices.SNOW_STORE] = 0.2
            store_outflow_weights[self.Indices.SURFACE_STORE] = 1

            self.outflow_weights[0, -self.Indices.STORE_DIM:] = store_outflow_weights

        scale_b = True
        hidden_dim = 100
        output_dim = 1
        num_layers = 2
        flow_between_stores = False  #Allow flow between stores; otherwise they're all connected only to out flow

        def b_length(self):
            return self.Indices.STORE_DIM * (self.Indices.STORE_DIM + 2) if self.flow_between_stores \
                else self.Indices.STORE_DIM

        def store_idx_start(self):
            return self.b_length()-self.Indices.STORE_DIM

        #should this come from the dataset properties or the datapoint?
        #def input_dim(self, dataset_properties: DatasetProperties):
        #    return dataset_properties.climate_norm.si
        def input_dim1(datapoint: DataPoint, encoding_dim: int, store_size: int):
            return datapoint.climate_data.shape[1] + datapoint.signatures.shape[1] + datapoint.attributes.shape[1] + \
                   encoding_dim + store_size
        def input_dim2(dataset_properties: DatasetProperties, encoding_dim: int, store_size: int):
            return len(dataset_properties.climate_norm) + len(dataset_properties.sig_normalizers) \
                + len(dataset_properties.attrib_normalizers) + encoding_dim + store_size

        def select_input(datapoints: DataPoint, encoding, stores):
            #raise Exception("todotodo check all these shapes") # t x i x b in caller
            batchsize=datapoints.climate_data.shape[2]
            timesteps=datapoints.climate_data.shape[0]
            decoder_input_dim = DecoderProperties.HydModelNetProperties.input_dim1(datapoints, encoding.shape[1], stores.shape[1])
            #decoder_input_dim2 = DecoderProperties.HydModelNetProperties.input_dim1(dataset_properties, encoding.shape[1],
            #if decoder_input_dim != decoder_input_dim2:
            #    raise Exception(f"Error calculating decoder_input_dim in 2 different ways decoder_input_dim="
            #                    f"{decoder_input_dim} decoder_input_dim2={decoder_input_dim2}")
            num_climate_attribs = datapoints.climate_data.shape[1]
            num_signatures = datapoints.signatures.shape[1]
            num_attributes = datapoints.attributes.shape[1]
            encoding_dim = encoding.shape[1]
            store_size = stores.shape[1]
            climate_start_idx=0
            sig_start_idx = climate_start_idx+num_climate_attribs
            attrib_start_idx = sig_start_idx+num_signatures
            encoding_start_idx = attrib_start_idx+num_attributes
            store_start_idx = encoding_start_idx+encoding_dim

            decoder_input = np.full([timesteps, decoder_input_dim, batchsize], np.nan)
            decoder_input[:, climate_start_idx:sig_start_idx, :] = datapoints.climate_data
            decoder_input[:, sig_start_idx:attrib_start_idx, :] = np.expand_dims(np.transpose(np.array(datapoints.signatures)), 0)
            decoder_input[:, attrib_start_idx:encoding_start_idx, :] = np.expand_dims(np.transpose(np.array(datapoints.attributes)), 0)
            decoder_input[:, encoding_start_idx:store_start_idx, :] = encoding
            decoder_input[:, store_start_idx:(store_start_idx+store_size), :] = stores
            if np.isnan(decoder_input).any().any():
                raise Exception("Failed to fill decoder_input")

            return torch.from_numpy(decoder_input)

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


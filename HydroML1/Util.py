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
    years_per_sample = 1
    length_days: int = 365 * years_per_sample

    #  attribs: dict of attrib/normalizing factor for each site
    attrib_normalizers = {'carbonate_rocks_frac': 1.0,  # Numbers, not categories, from geol.
               'geol_porostiy': 1.0,
               'geol_permeability': 0.1,
               'soil_depth_pelletier': 0.02,  # everything from soil
               'soil_depth_statsgo': 1,
               'soil_porosity': 0.2,
               'soil_conductivity': 0.2,
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
        'q95': 0.2,
        'high_q_freq': 0.05,
        'high_q_dur': 0.02,
        'low_q_freq': 0.01,
        'low_q_dur': 0.01,
        'zero_q_freq': 0.05,
        'hfd_mean': 0.01,
    }

    climate_norm = {
        'dayl(s)': 0.00002,
        'prcp(mm/day)': 0.05,
        'srad(W/m2)': 0.005,
        'swe(mm)': 1,
        'tmax(C)': 0.1,
        'tmin(C)': 0.1,
        'vp(Pa)': 0.001,
    }

    def num_sigs(self): return len(self.sig_normalizers)

    def sig_index(self, name): return list(self.sig_normalizers.keys()).index(name)

    def temperatures(self, datapoint: DataPoint):
        temps = np.zeros([datapoint.batch_size(), 2, datapoint.timesteps()])
        idx = get_indices(['tmin(C)'], datapoint.climate_data_cols)[0]
        temps[:, 0, :] = datapoint.climate_data[:, :, idx] / self.climate_norm['tmin(C)']
        idx = get_indices(['tmax(C)'], datapoint.climate_data_cols)[0]
        temps[:, 1, :] = datapoint.climate_data[:, :, idx] / self.climate_norm['tmax(C)']
        return temps

    def get_prob_rain(self, datapoint: DataPoint):
        is_rain = self.get_rain_np(datapoint) > 0.1
        return np.sum(is_rain, axis=1)/is_rain.shape[1]

    def runoff_ratio(self, datapoint: DataPoint):
        rr=self.get_sig(datapoint, 'runoff_ratio')
        if rr.min() <= 0 or rr.max() >= 1.5:
            raise Exception(f"Runoff ratio outside reasonable range min={rr.min()} max={rr.max()}")
        return rr

    def get_sig(self, datapoint: DataPoint, sig: str):
        #rr = np.zeros([datapoint.signatures.shape[0]])  #  # batches
        idx = get_indices([sig], datapoint.signatures)[0]
        return datapoint.signatures[sig].to_numpy() / self.sig_normalizers[sig]

    def get_rain_np(self, datapoints: DataPoint):
        return self.get_rain(datapoints).numpy()

    def get_rain(self, datapoints: DataPoint):
        idx_rain = get_indices(['prcp(mm/day)'], self.climate_norm.keys())[0]
        return datapoints.climate_data[:, :, idx_rain]/self.climate_norm['prcp(mm/day)']  # rain is t x b


class EncoderProperties:
    encoder_type = EncType.CNNEncoder
    encoder_names = ["prcp(mm/day)", "tmax(C)"]  # "swe(mm)",'flow(cfs)',
    flow_normalizer = 1  # 0.1 is too low...
    # encoder_indices = get_indices(encoder_names, hyd_data_labels)
    # indices = list(hyd_data_labels).index()

    def encoder_input_dim(self):
        return len(self.encoder_names)+1  # +1 for flow

    encoding_num_layers = 2
    encoding_hidden_dim = 20
    encode_attributes = True

    def encoding_dim(self):
        return 0 if self.encoder_type == EncType.NoEncoder else 16

    #def select_one_encoder_inputs(self, datapoint: DataPoint):
    #    datapoint.hydro_data: pd.DataFrame
    #    return np.array(datapoint.hydro_data(self.encoder_names))

    def encoder_perm(self):
        if self.encoder_type == EncType.LSTMEncoder:
            return (0, 2, 1)  # t x i x b -> t x b x i
        elif self.encoder_type == EncType.CNNEncoder:
            return (2, 1, 0)  # t x i x b -> b x i x t
        else:
            raise Exception("Encoder disabled or case not handled")

    def select_encoder_inputs(self, datapoint: DataPoint, dataset_properties: DatasetProperties):
        indices = get_indices(self.encoder_names, dataset_properties.climate_norm.keys())
        hyd_data = torch.cat((datapoint.flow_data*self.flow_normalizer, datapoint.climate_data[:, :, indices]), dim=2)\
            .permute(0, 2, 1)  # i x t x b -> b x i x t
        fixed_data = None if not self.encode_attributes \
            else torch.cat((torch.tensor(np.array(datapoint.signatures)), # match with encoding_names()
                            torch.tensor(np.array(datapoint.attributes))), 1)
        return (hyd_data, fixed_data)

    def encoding_names(dataset_properties: DatasetProperties):
        return list(dataset_properties.sig_normalizers.keys()) + list(dataset_properties.attrib_normalizers.keys())


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
            STORE_DIM = 8  # 4 is probably the minimum: snow, deep, shallow, runoff
            SLOW_STORE = 0
            SLOW_STORE2 = 1 # a bit faster
            SURFACE_STORE = 2
            SNOW_STORE = STORE_DIM-1

        def __init__(self):
            #store_weights normalize stores when used as input [UNUSED--log them instead]
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

        scale_b = False
        hidden_dim = 128
        flownet_intermediate_output_dim = 20
        num_layers = 4
        flow_between_stores = False  #Allow flow between stores; otherwise they're all connected only to out flow
        decoder_include_stores = True
        decoder_include_fixed = False

        def b_length(self):
            return self.Indices.STORE_DIM * (self.Indices.STORE_DIM + 2) if self.flow_between_stores \
                else self.Indices.STORE_DIM

        def store_idx_start(self):
            return self.b_length()-self.Indices.STORE_DIM

        def input_dim2(self, dataset_properties: DatasetProperties, encoding_dim: int):
            total_dim = len(dataset_properties.climate_norm) + encoding_dim
            if self.decoder_include_fixed:
                total_dim += len(dataset_properties.sig_normalizers) + len(dataset_properties.attrib_normalizers)
            if self.decoder_include_stores:
                total_dim += self.Indices.STORE_DIM
            return total_dim

        def select_input(self, datapoints: DataPoint, encoding, stores, dataset_properties: DatasetProperties):
            batchsize=datapoints.climate_data.shape[0]
            timesteps=datapoints.climate_data.shape[1]
            decoder_input_dim = self.input_dim2(dataset_properties, encoding.shape[1])
            num_climate_attribs = datapoints.climate_data.shape[2]

            num_signatures = datapoints.signatures.shape[1] if self.decoder_include_fixed else 0
            num_attributes = datapoints.attributes.shape[1] if self.decoder_include_fixed else 0

            encoding_dim = encoding.shape[1]
            store_size = stores.shape[1]
            climate_start_idx=0
            sig_start_idx = climate_start_idx+num_climate_attribs
            attrib_start_idx = sig_start_idx+num_signatures
            encoding_start_idx = attrib_start_idx+num_attributes
            store_start_idx = encoding_start_idx+encoding_dim

            decoder_input = torch.empty(timesteps, decoder_input_dim, batchsize, dtype=torch.double).fill_(np.nan)
            decoder_input[:, climate_start_idx:sig_start_idx, :] = datapoints.climate_data.permute(1, 2, 0)

            if self.decoder_include_fixed:
                decoder_input[:, sig_start_idx:attrib_start_idx, :] = np.expand_dims(np.transpose(np.array(datapoints.signatures)), 0)
                decoder_input[:, attrib_start_idx:encoding_start_idx, :] = np.expand_dims(np.transpose(np.array(datapoints.attributes)), 0)

            decoder_input[:, encoding_start_idx:store_start_idx, :] = torch.from_numpy(encoding) # TODO don't do this broadcast

            if self.decoder_include_stores:
                decoder_input[:, store_start_idx:(store_start_idx+store_size), :] = stores
            if np.isnan(decoder_input).any().any():
                raise Exception("Failed to fill decoder_input")

            return decoder_input

    hyd_model_net_props = HydModelNetProperties()


def get_indices(encoder_names, hyd_data_labels):
    indices = [i for i, x in enumerate(hyd_data_labels) if x in encoder_names]
    if len(indices) != len(encoder_names):
        raise Exception()
    return indices


def print_inputs(name, hyd_data):
    hyd_data.max().max()
    print(name + f' inputs max {torch.max(torch.max(hyd_data))}, mean {torch.mean(torch.mean(hyd_data))}')


dropout_rate = 0 # 0.33
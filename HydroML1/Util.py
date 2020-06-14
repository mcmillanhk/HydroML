from enum import Enum


class EncType(Enum):
    NoEncoder = 0
    LSTMEncoder = 1
    CNNEncoder = 2  # not implemented yet


def get_indices(encoder_names, hyd_data_labels):
    indices = [i for i, x in enumerate(hyd_data_labels) if x in encoder_names]
    #indices = []
    #for i, x in enumerate(hyd_data_labels):
    #    if x in encoder_names:
    #        indices = [indices, i]
    if len(indices) != len(encoder_names):
        raise Exception()
    return indices


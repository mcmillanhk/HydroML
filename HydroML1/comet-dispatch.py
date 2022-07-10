# Script for hyperparameter training on cluster
import argparse

from Hyd_ML import train_test_everything, plotting_freq, batch_size, interstore_weight_eps, weight_decay
import sys

from Util import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train with a set of hyperparameters')
    parser.add_argument('--flow_between_stores', type=bool, nargs='?', default=True,
                        help='Whether to model flow_between_stores')
    parser.add_argument('--num_stores', type=int, default=24, nargs='?',
                        help='')
    parser.add_argument('--reload', type=int, default=0, nargs='?',
                        help='Whether to reload the last model from the same directory (E200)')
    parser.add_argument('--log_batch_size', type=int, nargs='?', default=None)
    parser.add_argument('--years_per_sample', type=int, nargs='?', default=1)
    parser.add_argument('--interstore_weight_eps', type=int, nargs='?', default=4)
    parser.add_argument('--weight_decay', type=int, nargs='?', default=1)

    args = parser.parse_args()

    global plotting_freq
    plotting_freq = 0

    global batch_size
    if args.log_batch_size is not None:
        batch_size = int(2 ** args.log_batch_size)

    global interstore_weight_eps
    interstore_weight_eps = 0.005 * (args.interstore_weight_eps-1)

    global weight_decay
    weight_decay = 0.005 * (args.weight_decay-1)

    encoder_properties = EncoderProperties()
    decoder_properties = DecoderProperties()
    decoder_properties.hyd_model_net_props.flow_between_stores = args.flow_between_stores
    decoder_properties.hyd_model_net_props.store_dim = args.num_stores

    train_test_everything(1, 1, r"/cw3e/mead/projects/cwp101/scratch/hilarymcmillan/camels-us/basin_dataset_public_v1p2",
                          'models/Epoch200' if args.reload else None, 'models', data_root=r"/home/hilarymcmillan/hydro/HydroML/data",
                          encoder_properties=encoder_properties, decoder_properties=decoder_properties,
                          years_per_sample=args.years_per_sample)

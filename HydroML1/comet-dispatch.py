# Script for hyperparameter training on cluster
import argparse
import os.path

from Hyd_ML import train_test_everything, plotting_freq
import sys

from Util import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train with a set of hyperparameters')
    parser.add_argument('--flow_between_stores', type=bool, nargs='?', default=None,
                        help='Whether to model flow_between_stores')
    parser.add_argument('--num_stores', type=int, default=None, nargs='?',
                        help='')
    parser.add_argument('--encoding_dim', type=int, default=None, nargs='?',
                        help='')
    parser.add_argument('--reload', type=int, default=0, nargs='?',
                        help='Whether to reload the last model from the same directory (E200)')
    parser.add_argument('--log_batch_size', type=int, nargs='?', default=None)
    parser.add_argument('--years_per_sample', type=int, nargs='?', default=None)
    parser.add_argument('--encoder_years_per_sample', type=int, nargs='?', default=None)
    parser.add_argument('--decoder_years_per_sample', type=int, nargs='?', default=None)
    parser.add_argument('--newman_split', type=bool, nargs='?', default=None)
    parser.add_argument('--interstore_weight_eps', type=int, nargs='?', default=None)
    parser.add_argument('--weight_decay', type=int, nargs='?', default=None)
    parser.add_argument('--lr', type=int, nargs='?', default=None)
    parser.add_argument('--huber_thresh', type=int, nargs='?', default=None)
    parser.add_argument('--bn_eps', type=int, nargs='?', default=None)
    parser.add_argument('--bn_momentum', type=int, nargs='?', default=None)

    args = parser.parse_args()

    global plotting_freq
    plotting_freq = 0

    training_properties = TrainingProperties()
    if args.log_batch_size is not None:
        training_properties.batch_size = int(2 ** args.log_batch_size)

    if args.interstore_weight_eps is not None:
        training_properties.interstore_weight_eps = 0.005 * (args.interstore_weight_eps-1)

    if args.weight_decay is not None:
        training_properties.weight_decay = 0.005 * (args.weight_decay-1)

    if args.lr is not None:
        training_properties.learning_rate = 0.0005 * args.lr

    if args.huber_thresh is not None:
        training_properties.huber_thresh = 0.05 * args.huber_thresh

    encoder_properties = EncoderProperties()
    decoder_properties = DecoderProperties()
    if args.flow_between_stores is not None:
        decoder_properties.hyd_model_net_props.flow_between_stores = args.flow_between_stores
    if args.num_stores is not None:
        decoder_properties.hyd_model_net_props.store_dim = args.num_stores
    if args.encoding_dim is not None:
        encoder_properties.hydro_encoding_output_dim = args.encoding_dim

    dataloader_properties = DataloaderProperties()
    if args.years_per_sample is not None:
        dataloader_properties.decoder_years_per_sample = args.years_per_sample
        dataloader_properties.encoder_years_per_sample = args.years_per_sample

    if args.encoder_years_per_sample is not None:
        dataloader_properties.encoder_years_per_sample = args.encoder_years_per_sample
    if args.decoder_years_per_sample is not None:
        dataloader_properties.decoder_years_per_sample = args.decoder_years_per_sample

    if args.newman_split is not None:
        dataloader_properties.newman_split = args.newman_split

    bn_params = BNParams()
    if args.bn_eps is not None:
        bn_params.eps = 0.01 * (2 ** args.bn_eps)
    if args.bn_momentum is not None:
        bn_params.momentum = None if args.bn_momentum == 1 else 0.01 * (2 ** args.bn_eps)
    encoder_properties.bn_params = bn_params
    decoder_properties.hyd_model_net_props.bn_params = bn_params

    path = None
    if args.reload:  # Find the most trained model to reload
        for i in range(1000, 0, -1):
            path = f"models/Epoch{i*100}"  # Must divide save_frequency
            if os.path.exists(path):
                print("Reload from " + path)
                break

    train_test_everything(1, r"/cw3e/mead/projects/cwp101/scratch/hilarymcmillan/camels-us/basin_dataset_public_v1p2",
                          path, 'models', data_root=r"/home/hilarymcmillan/hydro/HydroML/data",
                          encoder_properties=encoder_properties, decoder_properties=decoder_properties, training_properties=training_properties,
                          dataloader_properties=dataloader_properties)

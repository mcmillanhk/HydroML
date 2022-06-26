# Script for hyperparameter training on cluster
from Hyd_ML import train_test_everything
import sys

if __name__ == '__main__':
    #input_file = sys.argv[1]
    #search_algo_str = sys.argv[2]

    train_test_everything(10, 1, r"/cw3e/mead/projects/cwp101/scratch/hilarymcmillan/camels-us/basin_dataset_public_v1p2",
                          None, data_root=r"/home/hilarymcmillan/hydro/HydroML/data")

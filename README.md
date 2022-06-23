Encoder-decoder framework for jointly learning hydrologic signatures and streamflow.
By Tom Botterill (botterill.tom@gmail.com) and Hilary McMillan (hmcmillan@sdsu.edu)

First download and unzip CAMELS US dataset. Pass this as camels_path

Top-level functions are in Hyd_ML.py

from Hyd_ML import *

# See models directory for some pre-trained models.
# A saved model consists of 4 files: encoder.ckpt, encoder_properties.pkl, decoder.ckpt, decoder_properties.pkl. 

# To load a pre-trained model and generate the figures in the paper (plus many more comparing 2 models with different random initialization):
compare_models(r"C:\\hydro\\basin_dataset_public_v1p2", r"C:\\hydro\\HydroML\\data", 1,
               [(r"C:\\hydro\\HydroML\\models\\E16-S8-1", "Learn Signatures"),
                (r"C:\\hydro\\HydroML\\models\\E16-S8-2", "Learn Signatures2")])

# To train from random initialization, loading 1/10th of CAMELS catchments:
train_test_everything(10, 1, r"C:\\hydro\\basin_dataset_public_v1p2", None, data_root=r"C:\\hydro\\HydroML\\data")

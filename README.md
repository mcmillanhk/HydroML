## Encoder-decoder framework for jointly learning hydrologic signatures and streamflow.
By Tom Botterill (botterill.tom@gmail.com) and Hilary McMillan (hmcmillan@sdsu.edu)

First download, unzip, and organize. CAMELS US dataset from https://ral.ucar.edu/solutions/products/camels. We used CAMELS 1.2 and CAMELS CATCHMENT ATTRIBUTES 2.0.

### Download
Download "CAMELS time series meteorology, observed flow, meta data (.zip)"
Download "CAMELS CATCHMENT ATTRIBUTES"

### Unzip
```
unzip basin_timeseries_v1p2_modelOutput_daymet.zip
unzip camels_attributes_v2.0.zip
```

### Organize:
```
mv camels_attributes_v2.0 basin_dataset_public_v1p2
```
Pass camels_path=/your/path/to/camels_attributes_v2.0 


Top-level functions are in Hyd_ML.py

```
from Hyd_ML import *
```

See models directory for some pre-trained models.
A saved model consists of 4 files: encoder.ckpt, encoder_properties.pkl, decoder.ckpt, decoder_properties.pkl. 

To load a pre-trained model and generate the figures in the paper (plus many more comparing 2 models with different random initialization):
```
compare_models(r"C:\\hydro\\basin_dataset_public_v1p2", r"C:\\hydro\\HydroML\\data", 1,
               [(r"C:\\hydro\\HydroML\\models\\E16-S8-1", "Learn Signatures"),
                (r"C:\\hydro\\HydroML\\models\\E16-S8-2", "Learn Signatures2")])
```

To train from random initialization, loading 1/10th of CAMELS catchments:
```
train_test_everything(10, 1, r"C:\\hydro\\basin_dataset_public_v1p2", None, data_root=r"C:\\hydro\\HydroML\\data")
```

Also see:
do_ablation_test(): Load data from one catchment at a time and fit model (to test how good the model could perform, and whether the 
decoder structure is suitable.

can_encoder_learn_sigs(): Test whether/how well this encoder structure can learn existing CAMELS signatures.

analyse_one_site(): Run a single catchment with perturbed encodings, to test for their effect.
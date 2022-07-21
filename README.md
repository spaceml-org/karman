# karman-project

## Collect data:
The following folders & files shall be stored under the `data_directory` path:
* Thermospheric density, OMNIWeb, FISM2 Daily and FISM2 flare data must be stored, in the following folder: `jb08_nrlmsise_all_v1/`, `data_omniweb_v1/`, `fism2_daily_v1/`, `fism2_flare_v1/`
* `train_indices.txt`, `test_indices.txt`, `val_indices.txt`: these must be present only if the user wants to load indices from file, otherwise, they are created on the fly (you must pass the flag: `load_indices` as `True` for this to happen)

## Run the training script:

This will train a simple feed forward neural network without omni and fism2 flare data.
```
python scripts/train.py --data_directory /home/jupyter/data --model FeedForwardDensityPredictor --exclude_omni --exclude_fism2
```

This will train a model using an LSTM-based architecture to process the different data source independently and then concatenate features into a feed forward layer at the end.

```
python scripts/train.py --data_directory /home/jupyter/data --model FullFeatureDensityPredictor --lag_minutes_omni 10 --lag_days_fism2_daily 100 --lag_minutes_fism2_flare 100
```

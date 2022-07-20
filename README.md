# karman-project

## Run the training script

This will train a simple feed forward neural network without omni and fism2 flare data.
```
python scripts/train.py --data_directory /home/jupyter/data --model FeedForwardDensityPredictor --exclude_omni --exclude_fism2
```

This will train a model using an LSTM-based architecture to process the different data source independently and then concatenate features into a feed forward layer at the end.

```
python scripts/train.py --data_directory /home/jupyter/data --model FullFeatureDensityPredictor --lag_minutes_omni 10 --lag_days_fism2_daily 100 --lag_minutes_fism2_flare 100
```
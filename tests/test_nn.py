import unittest
import karman
import os
import torch
import pandas as pd
from io import StringIO

class NNTestCases(unittest.TestCase):
    def test_model_and_state_dict(self):    
        omni_resolution=1
        fism2_flare_stan_bands_resolution=1
        fism2_daily_stan_bands_resolution=60*24
        dataset=karman.ThermosphericDensityDataset(
            directory=os.path.join(os.getcwd(),"tests/data"),
            include_omni=True,
            include_daily_stan_bands=True,
            include_flare_stan_bands=True,
            thermo_scaler=None,
            min_date=pd.to_datetime('2004-02-01 00:00:00'),
            max_date=pd.to_datetime('2004-02-03 00:00:00'),
            lag_minutes_omni=0,
            lag_minutes_fism2_flare_stan_bands = 0,
            lag_minutes_fism2_daily_stan_bands = 0,
            create_cyclical_features=False,
            omni_resolution=omni_resolution,
            fism2_flare_stan_bands_resolution=fism2_flare_stan_bands_resolution,
            fism2_daily_stan_bands_resolution=fism2_daily_stan_bands_resolution

        )
        
        dataset._set_indices(test_month_idx=[1], validation_month_idx=[0])        
        model_path='tests/data/kml_model'

        model = karman.nn.SimpleNN().to(dtype=torch.float64)

        state_dict = torch.load(os.path.join(model_path),map_location=torch.device('cpu'))['state_dict']
        #Sanitize state_dict key names
        for key in list(state_dict.keys()):
            if key.startswith('module'):
                # Model was saved as dataparallel model
                # Remove 'module.' from start of key
                state_dict[key[7:]] = state_dict.pop(key)
            else:
                continue
        model.load_state_dict(state_dict)

        dummy_loader = torch.utils.data.DataLoader(dataset,       
                                                    batch_size=2,
                                                    pin_memory=False,
                                                    num_workers=0,
                                                    drop_last=True)
        #I check it works:
        next(iter(dummy_loader))

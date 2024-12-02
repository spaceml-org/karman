import torch
import pickle as pk
import pandas as pd
import numpy as np
import os
from tft_torch import tft
from omegaconf import OmegaConf
torch.set_default_dtype(torch.float32)

from . import nn
from . import util
from importlib.resources import files

#get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

#date_to_index, scale_density, util.get_normalized_time_series, normalize_time_series_data
#here we load the necessary scalers:
# Construct the path to the data file
scalers_dict={}
keys_time_series_data=['omni_indices', 'omni_solar_wind', 'omni_magnetic_field', 'soho', 'msise']
for key in keys_time_series_data:
    with files("karman").joinpath(f"scaler_{key}.pk").open("rb") as f:
        scalers_dict[key] = pk.load(f)
with files("karman").joinpath("normalization_dict_ts.pk").open("rb") as f:
    _normalization_dict_ts=pk.load(f)

with files("karman").joinpath("normalization_dict.pk").open("rb") as f:
    _normalization_dict=pk.load(f)

#we also load the data for the space-weather indices, in case needed:
file_path = files("karman").joinpath("satellites_data_subsampled_1d.csv")

#file_path = files("karman").joinpath("data/merged_datasets/satellites_data_subsampled_1d.csv")
df_sw=pd.read_csv(file_path)

class ForecastingModel():
    def __init__(self,
                 dropout=0.05,
                 state_size=64,
                 lstm_layers=2,
                 attention_heads=4,
                 num_static_features=8,
                 num_future_numeric=1,
                 normalization_dict_ts=_normalization_dict_ts
                 ):
                 
        #default values
        self.dropout=dropout
        self.state_size=state_size
        self.lstm_layers=lstm_layers
        self.attention_heads=attention_heads
        self.num_static_features=num_static_features
        self.num_future_numeric=num_future_numeric
        self._omni_indices={'lag':10000,'resolution': 100, 'num_features': 6}
        self._omni_solar_wind={'lag':10000,'resolution': 100, 'num_features': 4}
        self._omni_magnetic_field={'lag':10000,'resolution': 100, 'num_features': 3}
        self._soho={'lag':10000,'resolution': 100, 'num_features': 2}
        self._msise={'lag':10000,'resolution': 100, 'num_features': 10}
        self.update_tft_configuration()
        #now for the forecasting part
        self._normalization_dict_ts=normalization_dict_ts


    def compute_data_props(self,num_static_features=8,num_future_numeric=1):
        #TFT parameters:
        num_historical_numeric=0
        num_historical_numeric+=self._omni_indices['num_features']#karman_dataset[0]['omni_indices'].shape[1]
        num_historical_numeric+=self._omni_magnetic_field['num_features']#karman_dataset[0]['omni_magnetic_field'].shape[1]
        num_historical_numeric+=self._omni_solar_wind['num_features']#karman_dataset[0]['omni_solar_wind'].shape[1]
        num_historical_numeric+=self._msise['num_features']#karman_dataset[0]['msise'].shape[1]
        num_historical_numeric+=self._soho['num_features']#karman_dataset[0]['soho'].shape[1]
        data_props = {'num_historical_numeric': num_historical_numeric,
                      'num_static_numeric': num_static_features,
                      'num_future_numeric': num_future_numeric,
                     }
        return data_props

    @property
    def omni_indices(self):
        return self._omni_indices
    @property
    def omni_solar_wind(self):
        return self._omni_solar_wind
    @property
    def omni_magnetic_field(self):
        return self._omni_magnetic_field
    @property
    def soho(self):
        return self._soho
    @property
    def msise(self):
        return self._msise

    @omni_indices.setter
    def omni_indices(self, value):
        if isinstance(value, dict):
            self._omni_indices = value
            self.update_tft_configuration()
        else:
            raise ValueError("omni_indices must be a dictionary")
    @omni_solar_wind.setter
    def omni_solar_wind(self, value):
        if isinstance(value, dict):
            self._omni_solar_wind = value
            self.update_tft_configuration()
        else:
            raise ValueError("omni_solar_wind must be a dictionary")
    @omni_magnetic_field.setter
    def omni_magnetic_field(self, value):
        if isinstance(value, dict):
            self._omni_magnetic_field = value
            self.update_tft_configuration()
        else:
            raise ValueError("omni_magnetic_field must be a dictionary")
    @soho.setter
    def soho(self, value):
        if isinstance(value, dict):
            self._soho = value
            self.update_tft_configuration()
        else:
            raise ValueError("soho must be a dictionary")
    @msise.setter
    def msise(self, value):
        if isinstance(value, dict):
            self._msise = value
            self.update_tft_configuration()
        else:
            raise ValueError("msise must be a dictionary")
    
    def update_tft_configuration(self):
        data_props=self.compute_data_props(num_static_features=self.num_static_features,
                                      num_future_numeric=self.num_future_numeric)
        self._tft_configuration = {'model':
                                {
                                    'dropout': self.dropout,
                                    'state_size': self.state_size,
                                    'output_quantiles': [0.5],
                                    'lstm_layers': self.lstm_layers,
                                    'attention_heads': self.attention_heads,
                                },
                            'task_type': 'regression',
                            'target_window_start': None,
                            'data_props': data_props,
                            }
    def load_model(self,model_path,device='cpu'):
        ts_karman_model = tft.TemporalFusionTransformer(OmegaConf.create(self._tft_configuration))
        ts_karman_model.apply(nn.weight_init)
        ts_karman_model.to(device)
        num_params=sum(p.numel() for p in ts_karman_model.parameters() if p.requires_grad)
        print(f"number of parameters: {num_params}")
        ts_karman_model.load_state_dict(torch.load(model_path,map_location=torch.device(device)))
        self._loaded_model = ts_karman_model.eval()  # Store the model
        print(f"Loaded model with {sum(p.numel() for p in ts_karman_model.parameters() if p.requires_grad)} parameters")


    def __call__(self,
                 dates, 
                 altitudes,
                 longitudes,
                 latitudes,
                 ts_data_normalized,
                 df_thermo=df_sw,
                 device=torch.device('cpu')):
        normalization_dict_ts=self._normalization_dict_ts      
        static_features = util.get_static_data(df_thermo=df_thermo,
                                                dates=dates,
                                                normalization_dict=normalization_dict_ts,
                                                altitudes=altitudes,
                                                longitudes=longitudes,
                                                latitudes=latitudes)

        historical_ts_numeric, future_ts_numeric = util.get_ts_data(
                                                                    ts_data_normalized, dates,
                                                                    omni_indices=self._omni_indices,
                                                                    omni_solar_wind=self._omni_solar_wind,
                                                                    omni_magnetic_field=self._omni_magnetic_field,
                                                                    soho=self._soho,
                                                                    msise=self._msise
                                                                )

        inputs = {
                  'static_feats_numeric': static_features.to(device),
                  'future_ts_numeric': future_ts_numeric.to(device),
                  'historical_ts_numeric': historical_ts_numeric.to(device),
                 }

        if self._loaded_model is None:
            raise RuntimeError("Model is not loaded. Please call load_model() first.")

        with torch.no_grad():
            batch_out = self._loaded_model(inputs)

        predicted_quantiles = batch_out['predicted_quantiles']  # Shape: [batch_size x future_steps x num_quantiles]
        target_nn_median = predicted_quantiles[:, :, 0].squeeze()

        log_min = normalization_dict_ts['log_density']['min']
        log_max = normalization_dict_ts['log_density']['max']
        tmp = (log_max - log_min) * (target_nn_median.detach().cpu() + 1) / 2 + log_min
        rho_nn = torch.pow(10, tmp)
        return rho_nn.numpy()

class NowcastingModel():
    def __init__(self,
                 num_instantaneous_features=18,
                 hidden_layer_dims=128,
                 hidden_layers=3,
                 normalization_dict=_normalization_dict,
                 df_sw=df_sw
                 ):
        self.num_instantaneous_features=num_instantaneous_features
        self.hidden_layer_dims=hidden_layer_dims
        self.hidden_layers=hidden_layers
        self._normalization_dict=normalization_dict
        self._df_sw=df_sw

    def load_model(self,model_path,device='cpu'):
        device=torch.device(device)    
        karman_model=nn.SimpleNetwork( input_dim=self.num_instantaneous_features,
                                            act=torch.nn.LeakyReLU(negative_slope=0.01),
                                            hidden_layer_dims=[self.hidden_layer_dims]*self.hidden_layers,
                                            output_dim=1).to(device)
        karman_model.load_state_dict(torch.load(model_path))
        self._loaded_model = karman_model  # Store the model
        print(f"Loaded model with {sum(p.numel() for p in karman_model.parameters() if p.requires_grad)} parameters")

    def __call__(self,
                 dates, 
                 altitudes,
                 longitudes,
                 latitudes,
                 device=torch.device('cpu'),
                 df_thermo=None,
                 normalization_dict=None
                      ):
        """
        Performs the inference of the nowcasting model.

        Args:
            - dates (`list`): list of datetime objects
            - altitudes (`list`): list of altitudes (in meters)
            - longitudes (`list`): list of longitudes (in degrees, between -180-180)
            - latitudes (`list`): list of latitudes (in degrees, between -90, 90)
            - df_thermo (`pd.DataFrame`): pandas dataframe containing instantaneous features
            - karman_model (`torch.nn.Module`): the nowcasting model
            - device (`str`): device to run the model on ('cpu' or 'cuda')
        
        Returns:
            - `np.array` predicted density (in kg/m**3)
        """
        if df_thermo is None:    
            df_thermo=self._df_sw
        if normalization_dict is None:    
            normalization_dict=self._normalization_dict

        static_features = util.get_static_data(df_thermo=df_thermo,
                                                dates=dates,
                                                normalization_dict=normalization_dict,
                                                altitudes=altitudes,
                                                longitudes=longitudes,
                                                latitudes=latitudes)

        if self._loaded_model is None:
            raise RuntimeError("Model is not loaded. Please call load_model() first.")

        expo=util.exponential_atmosphere(torch.tensor(altitudes)/1e3)
        with torch.no_grad():
            out_nn=torch.tanh(self._loaded_model(static_features).flatten())
        #scale the exponential density:
        tmp = torch.log10(expo)
        log_min=normalization_dict['log_density']['min']
        log_max=normalization_dict['log_density']['max']
        scaled_expo=2. * (tmp - log_min) / (log_max - log_min) - 1.
        out=scaled_expo+out_nn
        #unscale the output:
        tmp=(log_max-log_min)*(out+1)/2+log_min    
        return torch.pow(10,tmp).numpy()

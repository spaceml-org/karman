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

#get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

#date_to_index, scale_density, util.get_normalized_time_series, normalize_time_series_data
#here we load the necessary scalers:
# Construct the path to the data file
scalers_dict={}
keys_time_series_data=['omni_indices', 'omni_solar_wind', 'omni_magnetic_field', 'soho', 'msise']
for key in keys_time_series_data:
    with open(os.path.join(script_dir,f"scaler_{key}.pk"),"rb") as f:
        scalers_dict[key]=pk.load(f)

with open(os.path.join(script_dir,"normalization_dict_ts.pk"), "rb") as f:
    _normalization_dict_ts=pk.load(f)

with open(os.path.join(script_dir,"normalization_dict.pk"), "rb") as f:
    _normalization_dict=pk.load(f)

#we also load the data for the space-weather indices, in case needed:
df_sw=pd.read_csv(os.path.join(script_dir,'../data/merged_datasets/satellites_data_subsampled_1d.csv'))

#default values
_omni_indices={'lag':10000,'resolution': 100, 'num_features': 6}
_omni_solar_wind={'lag':10000,'resolution': 100, 'num_features': 4}
_omni_magnetic_field={'lag':10000,'resolution': 100, 'num_features': 3}
_soho={'lag':10000,'resolution': 100, 'num_features': 2}
_msise={'lag':10000,'resolution': 100, 'num_features': 10}

#TFT parameters:
num_historical_numeric=0
num_historical_numeric+=_omni_indices['num_features']#karman_dataset[0]['omni_indices'].shape[1]
num_historical_numeric+=_omni_magnetic_field['num_features']#karman_dataset[0]['omni_magnetic_field'].shape[1]
num_historical_numeric+=_omni_solar_wind['num_features']#karman_dataset[0]['omni_solar_wind'].shape[1]
num_historical_numeric+=_msise['num_features']#karman_dataset[0]['msise'].shape[1]
num_historical_numeric+=_soho['num_features']#karman_dataset[0]['soho'].shape[1]
num_static_features=8#len()
data_props = {'num_historical_numeric': num_historical_numeric,
            'num_static_numeric': num_static_features,
            'num_future_numeric': 1,
            }

_tft_configuration = {'model':
                        {
                            'dropout': 0.05,
                            'state_size': 64,
                            'output_quantiles': [0.5],
                            'lstm_layers': 2,
                            'attention_heads': 4,
                        },
                    'task_type': 'regression',
                    'target_window_start': None,
                    'data_props': data_props,
                    }

def load_model(num_instantaneous_features,
               model_path=None,
               device="cpu",
               prediction_type="nowcasting",
               hidden_layer_dims=128,
               hidden_layers=3,
               configuration=_tft_configuration,
               ):
    
    if prediction_type=="nowcasting":
        device=torch.device(device)    
        karman_model=nn.SimpleNetwork( input_dim=num_instantaneous_features,
                                            act=torch.nn.LeakyReLU(negative_slope=0.01),
                                            hidden_layer_dims=[hidden_layer_dims]*hidden_layers,
                                            output_dim=1).to(device)
        karman_model.load_state_dict(torch.load(model_path))
        num_params=sum(p.numel() for p in karman_model.parameters() if p.requires_grad)
        print(f"number of parmaeters: {num_params}")
        return karman_model
    elif prediction_type=="forecasting":
        ts_karman_model = tft.TemporalFusionTransformer(OmegaConf.create(configuration))
        ts_karman_model.apply(nn.weight_init)
        ts_karman_model.to(device)
        num_params=sum(p.numel() for p in ts_karman_model.parameters() if p.requires_grad)
        print(f"number of parameters: {num_params}")
        ts_karman_model.load_state_dict(torch.load(model_path,map_location=torch.device(device)))
        return ts_karman_model

def nowcasting_model(dates, 
                      altitudes,
                      longitudes,
                      latitudes,
                      karman_model,
                      df_thermo=None,
                      device=torch.device('cpu'),
                      normalization_dict=None
                      ):
#    def predict(dates, 
#            altitudes,
#            longitudes,
#            latitudes,
#            sw,
#            karman_model,
#            normalization_dict):
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
        df_thermo=df_sw
    if normalization_dict is None:
        normalization_dict=_normalization_dict
    static_features = util.get_static_data(df_thermo=df_sw,
                                            dates=dates,
                                            normalization_dict=normalization_dict,
                                            altitudes=altitudes,
                                            longitudes=longitudes,
                                            latitudes=latitudes)
    expo=util.exponential_atmosphere(torch.tensor(altitudes)/1e3)
    with torch.no_grad():
        out_nn=torch.tanh(karman_model(static_features).flatten())
    #scale the exponential density:
    tmp = torch.log10(expo)
    log_min=normalization_dict['log_density']['min']
    log_max=normalization_dict['log_density']['max']
    scaled_expo=2. * (tmp - log_min) / (log_max - log_min) - 1.
    out=scaled_expo+out_nn
    #unscale the output:
    tmp=(log_max-log_min)*(out+1)/2+log_min    
    return torch.pow(10,tmp).numpy()

def forecasting_model(dates, 
                      altitudes,
                      longitudes,
                      latitudes,
                      ts_karman_model,
                      ts_data_normalized,
                      normalization_dict_ts=None,
                      df_thermo=None,
                      device=torch.device('cpu'),
                      ):
    """
    Performs the inference of the nowcasting model.

    Args:
        - dates (`list`): list of datetime objects
        - altitudes (`list`): list of altitudes (in meters)
        - longitudes (`list`): list of longitudes (in degrees, between -180-180)
        - latitudes (`list`): list of latitudes (in degrees, between -90, 90)
        - df_thermo (`pd.DataFrame`): pandas dataframe containing instantaneous features
        - device (`str`): device to run the model on ('cpu' or 'cuda')
        - ts_karman_model (`torch.nn.Module`): time series model
        - ts_data_normalized (`dict`): dictionary containing the normalized time series data        
    """
    if df_thermo is None:
        df_thermo=df_sw
    if normalization_dict_ts is None:
        normalization_dict_ts=_normalization_dict_ts
    static_features = util.get_static_data(df_thermo=df_sw,
                                            dates=dates,
                                            normalization_dict=normalization_dict_ts,
                                            altitudes=altitudes,
                                            longitudes=longitudes,
                                            latitudes=latitudes)

    historical_ts_numeric, future_ts_numeric = util.get_ts_data(ts_data_normalized,
                                                                 dates)
    inputs={}
    inputs['static_feats_numeric']=static_features.to(device)
    inputs['future_ts_numeric']=future_ts_numeric.to(device)
    inputs['historical_ts_numeric']=historical_ts_numeric.to(device)
    with torch.no_grad():    
        batch_out=ts_karman_model(inputs)       
    #now the quantiles for the tft:
    predicted_quantiles = batch_out['predicted_quantiles']#it's of shape batch_size x future_steps x num_quantiles
    target_nn_median=predicted_quantiles[:, :, 0].squeeze()
    
    log_min=normalization_dict_ts['log_density']['min']
    log_max=normalization_dict_ts['log_density']['max']
    tmp=(log_max-log_min)*(target_nn_median.detach().cpu()+1)/2+log_min
    rho_nn = torch.pow(10,tmp)
    return rho_nn.numpy()
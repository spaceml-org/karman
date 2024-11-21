import torch
import pickle as pk
import pandas as pd
import numpy as np
from tft_torch import tft
from omegaconf import OmegaConf
torch.set_default_dtype(torch.float32)

from . import nn
from . import util
#date_to_index, scale_density, util.get_normalized_time_series, normalize_time_series_data
#here we load the necessary scalers:
scalers_dict={}
keys_time_series_data=['omni_indices', 'omni_solar_wind', 'omni_magnetic_field', 'soho', 'msise']
for key in keys_time_series_data:
    with open(f"scaler_{key}.pk","rb") as f:
        scalers_dict[key]=pk.load(f)

with open("normalization_dict_ts.pk", "rb") as f:
    normalization_dict_ts=pk.load(f)

with open("normalization_dict.pk", "rb") as f:
    normalization_dict=pk.load(f)

#we also load the data for the space-weather indices, in case needed:
df_sw=pd.read_csv('../data/data_subsampled_1d.csv')

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
                      df_thermo=df_sw,
                      device=torch.device('cpu'),
                      normalization_dict=normalization_dict
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
    dates=pd.to_datetime(dates)
    sw=util.find_sw_from_thermo(dates,df_thermo)
    dates=pd.to_datetime(dates)
    doy=torch.tensor(dates.dayofyear,dtype=torch.float32)
    #doy = date.timetuple().tm_yday
    sid = torch.tensor(dates.hour * 3600 + dates.minute * 60 + dates.second + dates.microsecond / 1e6,dtype=torch.float32)
    feature_as_radian = (2* np.pi * (doy-normalization_dict['all__day_of_year__[d]']["min"])/ (normalization_dict['all__day_of_year__[d]']["max"] - normalization_dict['all__day_of_year__[d]']["min"]))
    all_doy_sin=feature_as_radian.sin()
    all_doy_cos=feature_as_radian.cos()

    feature_as_radian = (2* np.pi * (sid-normalization_dict['all__seconds_in_day__[s]']["min"])/ (normalization_dict['all__seconds_in_day__[s]']["max"] - normalization_dict['all__seconds_in_day__[s]']["min"]))
    sid_sin=feature_as_radian.sin()
    sid_cos=feature_as_radian.cos()

    lon=torch.tensor(longitudes)
    lon_sin=torch.deg2rad(lon).sin()
    lon_cos=torch.deg2rad(lon).cos()

    inputs=[]
    alt_n=2*(torch.tensor(altitudes)-normalization_dict['tudelft_thermo__altitude__[m]']["min"])/(normalization_dict['tudelft_thermo__altitude__[m]']["max"]-normalization_dict['tudelft_thermo__altitude__[m]']["min"])-1
    lat_n=2*(torch.tensor(latitudes)-normalization_dict['tudelft_thermo__latitude__[deg]']["min"])/(normalization_dict['tudelft_thermo__latitude__[deg]']["max"]-normalization_dict['tudelft_thermo__latitude__[deg]']["min"])-1
    inputs.append(alt_n)
    inputs.append(lat_n)
    for feature in normalization_dict.keys():
        if feature not in ['all__day_of_year__[d]', 
                           'all__seconds_in_day__[s]', 
                           'tudelft_thermo__longitude__[deg]',
                           'tudelft_thermo__altitude__[m]',
                           'tudelft_thermo__latitude__[deg]',
                           'log_density']:
            try:
                #feature_sw=feature.split('__')[-2]
                inputs.append(2*(torch.tensor(sw[feature].values,dtype=torch.float32)-normalization_dict[feature]["min"])/(normalization_dict[feature]["max"]-normalization_dict[feature]["min"])-1)
            except Exception as e:
                print(f"Note, {feature} not found in the sw dictionary, you have to use: f107_obs, f107_average, s107_obs, s107_average, m107_obs, m107_average, f107_obs, f107_average, ap_average, d_st_dt")
    inputs.append(lon_sin)
    inputs.append(lon_cos)
    inputs.append(all_doy_sin)
    inputs.append(all_doy_cos)
    inputs.append(sid_sin)
    inputs.append(sid_cos)
    inputs=torch.stack(inputs,axis=1)
    expo=util.exponential_atmosphere(torch.tensor(altitudes)/1e3)
    with torch.no_grad():
        out_nn=torch.tanh(karman_model(inputs).flatten())
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
                      df_thermo=df_sw,
                      device=torch.device('cpu'),
                      omni_indices=_omni_indices,
                      omni_magnetic_field=_omni_magnetic_field,
                      omni_solar_wind=_omni_solar_wind,
                      soho=_soho,
                      msise=_msise):
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
        - omni_indices (`dict`): dictionary containing the parameters for the omni indices
        - omni_magnetic_field (`dict`): dictionary containing the parameters for the omni magnetic field
        - omni_solar_wind (`dict`): dictionary containing the parameters for the omni solar wind
        - soho (`dict`): dictionary containing the parameters for the SOHO data
        - msise (`dict`): dictionary containing the parameters for the NRLMSISE-00 data
        
    """
    dates=pd.to_datetime(dates)
    sw=util.find_sw_from_thermo(dates,df_thermo)

    doy=torch.tensor(dates.dayofyear,dtype=torch.float32)
    #doy = date.timetuple().tm_yday
    sid = torch.tensor(dates.hour * 3600 + dates.minute * 60 + dates.second + dates.microsecond / 1e6,dtype=torch.float32)
    feature_as_radian = (2* np.pi * (doy-normalization_dict_ts['all__day_of_year__[d]']["min"])/ (normalization_dict_ts['all__day_of_year__[d]']["max"] - normalization_dict_ts['all__day_of_year__[d]']["min"]))
    all_doy_sin=feature_as_radian.sin()
    all_doy_cos=feature_as_radian.cos()

    feature_as_radian = (2* np.pi * (sid-normalization_dict_ts['all__seconds_in_day__[s]']["min"])/ (normalization_dict_ts['all__seconds_in_day__[s]']["max"] - normalization_dict_ts['all__seconds_in_day__[s]']["min"]))
    sid_sin=feature_as_radian.sin()
    sid_cos=feature_as_radian.cos()

    lon=torch.tensor(longitudes,dtype=torch.float32)
    lon_sin=torch.deg2rad(lon).sin()
    lon_cos=torch.deg2rad(lon).cos()

    inputs={}
    static_features=[]
    alt_n=2*(torch.tensor(altitudes,dtype=torch.float32)-normalization_dict_ts['tudelft_thermo__altitude__[m]']["min"])/(normalization_dict_ts['tudelft_thermo__altitude__[m]']["max"]-normalization_dict_ts['tudelft_thermo__altitude__[m]']["min"])-1
    lat_n=2*(torch.tensor(latitudes,dtype=torch.float32)-normalization_dict_ts['tudelft_thermo__latitude__[deg]']["min"])/(normalization_dict_ts['tudelft_thermo__latitude__[deg]']["max"]-normalization_dict_ts['tudelft_thermo__latitude__[deg]']["min"])-1
    static_features.append(alt_n)
    static_features.append(lat_n)
    for feature in normalization_dict_ts.keys():
        if feature not in ['all__day_of_year__[d]', 
                           'all__seconds_in_day__[s]', 
                           'tudelft_thermo__longitude__[deg]',
                           'tudelft_thermo__altitude__[m]',
                           'tudelft_thermo__latitude__[deg]',
                           'log_density']:
            try:
                feature_sw=feature.split('__')[-2]
                static_features.append(2*(torch.tensor(sw[feature].values,dtype=torch.float32)-normalization_dict_ts[feature]["min"])/(normalization_dict_ts[feature]["max"]-normalization_dict_ts[feature]["min"])-1)
            except Exception as e:
                print(f"Note, {feature_sw} not found in the sw dictionary, you have to use: f107_obs, f107_average, s107_obs, s107_average, m107_obs, m107_average, f107_obs, f107_average, ap_average, d_st_dt")
    static_features.append(lon_sin)
    static_features.append(lon_cos)
    static_features.append(all_doy_sin)
    static_features.append(all_doy_cos)
    static_features.append(sid_sin)
    static_features.append(sid_cos)
    static_features=torch.stack(static_features,axis=1)
    inputs['static_feats_numeric']=static_features.to(device)

    historical_ts_numeric=[]
    #NOTE: historical_ts_numeric has shape: n_elements x n_time_steps x n_features
    #      future_ts_numeric has shape: n_elements x n_time_steps x n_features (we usually assume n_time_steps to be 1 here)
    #      static_feats_numeric has shape: n_elements x n_features
    if omni_indices["lag"] is not None and omni_indices["resolution"] is not None:
        #here I do this radomly, TODO -> do this via pulling from the database directly and then using the scaler to normalize:
        num_temporal=int(omni_indices["lag"]/omni_indices["resolution"])
        omni_indices_inputs=[]
        for date in dates:
            omni_indices_inputs.append(util.get_normalized_time_series(ts_data_normalized["omni_indices"]["values"],
                                                                pd.to_datetime(ts_data_normalized["omni_indices"]['start_date']),
                                                                resolution=omni_indices["resolution"],
                                                                lag=omni_indices["lag"],
                                                                date=date)[:-1,:])
        omni_indices_inputs=torch.cat(omni_indices_inputs).reshape((len(dates),num_temporal,ts_data_normalized["omni_indices"]["values"].shape[1]))
        historical_ts_numeric+=[omni_indices_inputs]

    if omni_magnetic_field["lag"] is not None and omni_magnetic_field["resolution"] is not None:
        num_temporal=int(omni_magnetic_field["lag"]/omni_magnetic_field["resolution"])
        omni_magnetic_field_inputs=[]
        for date in dates:
            omni_magnetic_field_inputs.append(util.get_normalized_time_series(ts_data_normalized["omni_magnetic_field"]["values"],
                                                                pd.to_datetime(ts_data_normalized["omni_magnetic_field"]['start_date']),
                                                                resolution=omni_magnetic_field["resolution"],
                                                                lag=omni_magnetic_field["lag"],
                                                                date=date)[:-1,:])
        omni_magnetic_field_inputs=torch.cat(omni_magnetic_field_inputs).reshape((len(dates),num_temporal,ts_data_normalized["omni_magnetic_field"]["values"].shape[1]))
        historical_ts_numeric+=[omni_magnetic_field_inputs]
    if omni_solar_wind["lag"] is not None and omni_solar_wind["resolution"] is not None:
        num_temporal=int(omni_solar_wind["lag"]/omni_solar_wind["resolution"])
        omni_solar_wind_inputs=[]
        for date in dates:
            omni_solar_wind_inputs.append(util.get_normalized_time_series(ts_data_normalized["omni_solar_wind"]["values"],
                                                                pd.to_datetime(ts_data_normalized["omni_solar_wind"]['start_date']),
                                                                resolution=omni_solar_wind["resolution"],
                                                                lag=omni_solar_wind["lag"],
                                                                date=date)[:-1,:])
        omni_solar_wind_inputs=torch.cat(omni_solar_wind_inputs).reshape((len(dates),num_temporal,ts_data_normalized["omni_solar_wind"]["values"].shape[1]))
        historical_ts_numeric+=[omni_solar_wind_inputs]

    if soho["lag"] is not None and soho["resolution"] is not None:
        num_temporal=int(soho["lag"]/soho["resolution"])
        soho_inputs=[]
        for date in dates:
            soho_inputs.append(util.get_normalized_time_series(ts_data_normalized["soho"]["values"],
                                                                pd.to_datetime(ts_data_normalized["soho"]['start_date']),
                                                                resolution=soho["resolution"],
                                                                lag=soho["lag"],
                                                                date=date)[:-1,:])
        soho_inputs=torch.cat(soho_inputs).reshape((len(dates),num_temporal,ts_data_normalized["soho"]["values"].shape[1]))
        historical_ts_numeric+=[soho_inputs]

    if msise["lag"] is not None and msise["resolution"] is not None:
        num_temporal=int(msise["lag"]/msise["resolution"])
        msise_inputs=[]
        msise_last=[]
        for date in dates:
            val=util.get_normalized_time_series(ts_data_normalized["msise"]["values"],
                                                                pd.to_datetime(ts_data_normalized["msise"]['start_date']),
                                                                resolution=msise["resolution"],
                                                                lag=msise["lag"],
                                                                date=date)
            msise_inputs.append(val[:-1,:])
            msise_last.append(val[-1,:])
        msise_inputs=torch.cat(msise_inputs).reshape((len(dates),num_temporal,ts_data_normalized["msise"]["values"].shape[1]))
        msise_last=torch.cat(msise_last).reshape((len(dates),1,ts_data_normalized["msise"]["values"].shape[1]))
        historical_ts_numeric+=[msise_inputs]

    if len(historical_ts_numeric)>1:
        historical_ts_numeric=torch.cat(historical_ts_numeric,dim=2)
    else:
        historical_ts_numeric=historical_ts_numeric[0]

    inputs['future_ts_numeric']=torch.tensor(np.random.rand(len(lon_sin),1,msise["num_features"]),dtype=torch.float32).to(device)
    inputs['historical_ts_numeric']=historical_ts_numeric.to(device)

    batch_out=ts_karman_model(inputs)       
    #now the quantiles for the tft:
    predicted_quantiles = batch_out['predicted_quantiles']#it's of shape batch_size x future_steps x num_quantiles
    target_nn_median=predicted_quantiles[:, :, 0].squeeze()
    
    log_min=normalization_dict_ts['log_density']['min']
    log_max=normalization_dict_ts['log_density']['max']
    tmp=(log_max-log_min)*(target_nn_median.detach().cpu()+1)/2+log_min
    rho_nn = torch.pow(10,tmp)
    return rho_nn.numpy()
import torch
import pandas as pd
import numpy as np
import math
from . import nn
from .density_models import scalers_dict, _normalization_dict, _normalization_dict_ts, df_sw, _omni_indices, _omni_magnetic_field, _omni_solar_wind, _soho, _msise

def exponential_atmosphere(altitudes):    
    """
    Compute the density of the atmosphere using the piecewise exponential model. 
    Reference: Vallado, D. A. (2013). Fundamentals of astrodynamics and applications (4th Edition). Microcosm Press.

    Parameters:
        - altitudes (`torch.Tensor`): the altitudes at which to compute the density, [km].

    Returns:
        - rhos (`torch.Tensor`): the density of the atmosphere at the given altitudes
    """
    # Base altitude for the exponential atmospheric model, [km]
    zb = torch.tensor([0., 25, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120,
      130, 140, 150, 180, 200, 250, 300, 350, 400, 450,
      500, 600, 700, 800, 900, 1000])

    zb_expand = zb.clone().detach()
    # Set the two endpoints of base altitude to a small enough and a large enough value respectively for extrapolation calculations.
    zb_expand[0],zb_expand[-1] = -torch.inf,torch.inf

    # Nominal density for the exponential atmospheric model, [kg/m³]
    rhob = torch.tensor([1.225, 3.899e-2, 1.774e-2, 3.972e-3, 1.057e-3,
        3.206e-4, 8.770e-5, 1.905e-5, 3.396e-6, 5.297e-7,
        9.661e-8, 2.438e-8, 8.484e-9, 3.845e-9, 2.070e-9,
        5.464e-10, 2.789e-10, 7.248e-11, 2.418e-11,
        9.518e-12, 3.725e-12, 1.585e-12, 6.967e-13,
        1.454e-13, 3.614e-14, 1.170e-14, 5.245e-15,3.019e-15])

    # Scale height for the exponential atmospheric model, [km]
    ZS = torch.tensor([7.249, 6.349, 6.682, 7.554, 8.382, 7.714, 6.549,
      5.799, 5.382, 5.877, 7.263, 9.473, 12.636, 16.149,
      22.523, 29.740, 37.105, 45.546, 53.628, 53.298,
      58.515, 60.828, 63.822, 71.835, 88.667, 124.64,181.05, 268.00])

    bin_indices = torch.searchsorted(zb, altitudes, right=True) - 1

    rhos = rhob[bin_indices]*torch.exp(-(altitudes-zb_expand[bin_indices])/ZS[bin_indices])      
    return rhos

def scale_density(density,normalization_dict):
    tmp = torch.log10(density)
    log_min=normalization_dict['log_density']['min']
    log_max=normalization_dict['log_density']['max']
    return 2. * (tmp - log_min) / (log_max - log_min) - 1.

def date_to_index(dates, date_start, delta_seconds):
    """
    Converts one or multiple dates to their corresponding indices based on the start date and a fixed time interval.
    
    Parameters:
    ------------
        - dates (`datetime` or array-like of `datetime`): Single date or an array of dates to convert to indices.
        - date_start (`datetime`): The start date of the dataset.
        - delta_seconds (`int`): The equally spaced time interval in seconds.
        
    Returns:
    ------------
        - indices (`int` or `np.ndarray`): The index or indices corresponding to the provided date(s).
    """
    # Convert inputs to Pandas Timestamps for consistency
    if not isinstance(dates, pd.Series):
        dates = pd.Series(dates)
    
    # Calculate the time difference (in seconds) between each date and the start date
    delta_seconds_array = (dates - pd.Timestamp(date_start)).dt.total_seconds()
    
    # Compute indices by dividing the difference by delta_seconds
    indices = np.floor(delta_seconds_array / delta_seconds).astype(int)
    
    return indices if len(indices) > 1 else indices.iloc[0]

def get_static_data(dates,
                    normalization_dict,
                    altitudes,
                    longitudes,
                    latitudes,
                    df_thermo=None):
    if normalization_dict is None:
        normalization_dict=_normalization_dict_ts
    if df_thermo is None:
        df_thermo=df_sw
    dates=pd.to_datetime(dates)
    sw=find_sw_from_thermo(dates,df_thermo)

    doy=torch.tensor(dates.dayofyear,dtype=torch.float32)
    #doy = date.timetuple().tm_yday
    sid = torch.tensor(dates.hour * 3600 + dates.minute * 60 + dates.second + dates.microsecond / 1e6,dtype=torch.float32)
    feature_as_radian = (2* np.pi * (doy-normalization_dict['all__day_of_year__[d]']["min"])/ (normalization_dict['all__day_of_year__[d]']["max"] - normalization_dict['all__day_of_year__[d]']["min"]))
    all_doy_sin=feature_as_radian.sin()
    all_doy_cos=feature_as_radian.cos()

    feature_as_radian = (2* np.pi * (sid-normalization_dict['all__seconds_in_day__[s]']["min"])/ (normalization_dict['all__seconds_in_day__[s]']["max"] - normalization_dict['all__seconds_in_day__[s]']["min"]))
    sid_sin=feature_as_radian.sin()
    sid_cos=feature_as_radian.cos()

    lon=torch.tensor(longitudes,dtype=torch.float32)
    lon_sin=torch.deg2rad(lon).sin()
    lon_cos=torch.deg2rad(lon).cos()

    inputs={}
    static_features=[]
    alt_n=2*(torch.tensor(altitudes,dtype=torch.float32)-normalization_dict['tudelft_thermo__altitude__[m]']["min"])/(normalization_dict['tudelft_thermo__altitude__[m]']["max"]-normalization_dict['tudelft_thermo__altitude__[m]']["min"])-1
    lat_n=2*(torch.tensor(latitudes,dtype=torch.float32)-normalization_dict['tudelft_thermo__latitude__[deg]']["min"])/(normalization_dict['tudelft_thermo__latitude__[deg]']["max"]-normalization_dict['tudelft_thermo__latitude__[deg]']["min"])-1
    static_features.append(alt_n)
    static_features.append(lat_n)
    for feature in normalization_dict.keys():
        if feature not in ['all__day_of_year__[d]', 
                           'all__seconds_in_day__[s]', 
                           'tudelft_thermo__longitude__[deg]',
                           'tudelft_thermo__altitude__[m]',
                           'tudelft_thermo__latitude__[deg]',
                           'log_density']:
            try:
                feature_sw=feature.split('__')[-2]
                static_features.append(2*(torch.tensor(sw[feature].values,dtype=torch.float32)-normalization_dict[feature]["min"])/(normalization_dict[feature]["max"]-normalization_dict[feature]["min"])-1)
            except Exception as e:
                print(f"Note, {feature_sw} not found in the sw dictionary, you have to use: f107_obs, f107_average, s107_obs, s107_average, m107_obs, m107_average, f107_obs, f107_average, ap_average, d_st_dt")
    static_features.append(lon_sin)
    static_features.append(lon_cos)
    static_features.append(all_doy_sin)
    static_features.append(all_doy_cos)
    static_features.append(sid_sin)
    static_features.append(sid_cos)
    static_features=torch.stack(static_features,axis=1)
    return static_features

def prepare_ts_data(data,
                                start_date,
                                resolution,
                                lag,
                                dates
                               ):
    """
    This function takes the time series normalized torch tensor and returns them in a three-dimensional tensor format,
    already compatible for time series forecasting.

    Parameters:
    ------------
        - data (`torch.Tensor`): The time series data tensor.
        - start_date (`str`): The start date of the dataset.
        - resolution (`int`): The resolution of the data in minutes.
        - lag (`int`): The lag in minutes.
        - dates (`list`): The list of dates to extract the sequences from the tensor.

    Returns:
    ------------
        - sequences (`torch.Tensor`): The time series data tensor in the forecasting format.
    """
    now_index = date_to_index(pd.to_datetime(dates), 
                                   start_date,
                                    60 * resolution)
    lagged_index = date_to_index(pd.to_datetime(dates) - pd.Timedelta(minutes=lag),
                                        start_date,
                                        60 * resolution,)
    sequences_past = []
    sequences_future = []
    for i in range(len(now_index)):
        window = data[lagged_index[i]:now_index[i]]  # Extract values from tensor, up to the semi-last one (since we want to predict the last)
        sequences_past.append(window)
        sequences_future.append(data[now_index[i]])
    return torch.stack(sequences_past), torch.stack(sequences_future)
    
def get_ts_data( ts_data_normalized,   
                 dates,
                 omni_indices=None,
                 omni_magnetic_field=None,
                 omni_solar_wind=None,
                 soho=None,
                 msise=None):
    """
    
        - omni_indices (`dict`): dictionary containing the parameters for the omni indices
        - omni_magnetic_field (`dict`): dictionary containing the parameters for the omni magnetic field
        - omni_solar_wind (`dict`): dictionary containing the parameters for the omni solar wind
        - soho (`dict`): dictionary containing the parameters for the SOHO data
        - msise (`dict`): dictionary containing the parameters for the NRLMSISE-00 data

    """
    if omni_indices is None:
        omni_indices = _omni_indices  # Assign default here
    if omni_magnetic_field is None:
        omni_magnetic_field = _omni_magnetic_field
    if omni_solar_wind is None:
        omni_solar_wind = _omni_solar_wind
    if soho is None:
        soho = _soho
    if msise is None:
        msise = _msise

    historical_ts_numeric=[]
    future_ts_numeric=[]
    #NOTE: historical_ts_numeric has shape: n_elements x n_time_steps x n_features
    #      future_ts_numeric has shape: n_elements x n_time_steps x n_features (we usually assume n_time_steps to be 1 here)
    #      static_feats_numeric has shape: n_elements x n_features
    if omni_indices["lag"] is not None and omni_indices["resolution"] is not None:
        #here I do this radomly, TODO -> do this via pulling from the database directly and then using the scaler to normalize:
        omni_indices_inputs,_=prepare_ts_data(ts_data_normalized["omni_indices"]["values"],
                                                                start_date=pd.to_datetime(ts_data_normalized["omni_indices"]['start_date']),
                                                                resolution=omni_indices["resolution"],
                                                                lag=omni_indices["lag"],
                                                                dates=dates)
        historical_ts_numeric+=[omni_indices_inputs]

    if omni_magnetic_field["lag"] is not None and omni_magnetic_field["resolution"] is not None:
        omni_magnetic_field_inputs=[]
        omni_magnetic_field_inputs,_=prepare_ts_data(ts_data_normalized["omni_magnetic_field"]["values"],
                                                            pd.to_datetime(ts_data_normalized["omni_magnetic_field"]['start_date']),
                                                            resolution=omni_magnetic_field["resolution"],
                                                            lag=omni_magnetic_field["lag"],
                                                            dates=dates)
        historical_ts_numeric+=[omni_magnetic_field_inputs]
    if omni_solar_wind["lag"] is not None and omni_solar_wind["resolution"] is not None:
        omni_solar_wind_inputs,_=prepare_ts_data(ts_data_normalized["omni_solar_wind"]["values"],
                                                                pd.to_datetime(ts_data_normalized["omni_solar_wind"]['start_date']),
                                                                resolution=omni_solar_wind["resolution"],
                                                                lag=omni_solar_wind["lag"],
                                                                dates=dates)
        historical_ts_numeric+=[omni_solar_wind_inputs]

    if soho["lag"] is not None and soho["resolution"] is not None:
        soho_inputs,_=prepare_ts_data(ts_data_normalized["soho"]["values"],
                                                            pd.to_datetime(ts_data_normalized["soho"]['start_date']),
                                                            resolution=soho["resolution"],
                                                            lag=soho["lag"],
                                                            dates=dates)
        historical_ts_numeric+=[soho_inputs]

    if msise["lag"] is not None and msise["resolution"] is not None:
        msise_inputs, msise_last = prepare_ts_data(ts_data_normalized["msise"]["values"],
                                                 pd.to_datetime(ts_data_normalized["msise"]['start_date']),
                                                            resolution=msise["resolution"],
                                                            lag=msise["lag"],
                                                            dates=dates)
        historical_ts_numeric+=[msise_inputs]
        future_ts_numeric+=[msise_last.reshape(msise_last.shape[0],1,msise_last.shape[1])]

    if len(historical_ts_numeric)>1:
        historical_ts_numeric=torch.cat(historical_ts_numeric,dim=2)
    else:
        historical_ts_numeric=historical_ts_numeric[0]

    if len(future_ts_numeric)>1:
        future_ts_numeric=torch.cat(future_ts_numeric,dim=2)
    else:
        future_ts_numeric=future_ts_numeric[0]
    return historical_ts_numeric, future_ts_numeric

#getter for time series data:
def normalize_time_series_data(resolution,
                                data_path,
                                normalization_dict_ts=_normalization_dict_ts
#                                min_date=pd.to_datetime("2000-07-29 00:59:47"),
#                                max_date=pd.to_datetime("2024-05-31 23:59:32")
                         ):
        print(f"Normalizing time series data (note that it can take a couple of minutes)")
        data_names=data_path.keys()
        # Data loading:
        time_series_data={}
        for data_name in data_names:
            time_series_data[data_name] = {}
            if data_name in ["omni_indices", "omni_solar_wind", "omni_magnetic_field","goes_256nm","goes_284nm","goes_304nm","goes_1175nm","goes_1216nm","goes_1335nm","goes_1405nm","soho"]:
                data = pd.read_csv(data_path[data_name])
                # we now index the data by the datetime column, and sort it by the index. The reason is that it is then easier to resample
                start_date=data["all__dates_datetime__"].min()
                data.index = pd.to_datetime(data["all__dates_datetime__"])
                data.sort_index(inplace=True)
                # We exclude the columns that are not needed for the model.
                data = data.drop(columns=['all__dates_datetime__', 'source__gaps_flag__'], axis=1)
                # This is to remove significant outliers, such as the fism2 flare data which has 10^45 photons at one point. Regardless
                # of whther this is true or not, it severely affects the distribution.
                for column in data.columns:
                    quantile = data[column].quantile(0.998)
                    more_than = data[column] >= quantile
                    data.loc[more_than, column] = None
                # We replace NaNs and +/-inf by interpolating them away.
                data = data.replace([np.inf, -np.inf], None)
                data.ffill(inplace=True)
                # We resample the data to the chosen resolution. We use forward fill, to fill in the gaps. Another possibility is the mean.
                #time_series_data[data_name]['data'] = time_series_data[data_name]['data'].resample(f'{resolution}T').mean()
                data = (data.resample(f"{resolution}min").ffill())
                # We store the start date of the dataset, and the data matrix.
                values = data.values
                # We scale the data, and convert it to a torch tensor.
                values = scalers_dict[data_name].fit_transform(values)  # .astype(np.float32)
                values = torch.tensor(values,dtype=torch.float32).detach()
                time_series_data[data_name]['values']=values
                time_series_data[data_name]['start_date']=start_date
            elif data_name in ["msise"]:
                    data = pd.read_csv(data_path[data_name])
                    start_date=data["all__dates_datetime__"].min()
                    # we now index the data by the datetime column, and sort it by the index. The reason is that it is then easier to resample
                    data.index = pd.to_datetime(data["all__dates_datetime__"])
                    data.sort_index(inplace=True)
                    # We exclude the columns that are not needed for the model.
                    if 'source__gaps_flag__' in data.columns:
                         data = data.drop(columns=['all__dates_datetime__', 'source__gaps_flag__'], axis=1)
                    else:
                        data = data.drop(columns=['all__dates_datetime__'], axis=1)
                    # We resample the data to the chosen resolution. We use forward fill, to fill in the gaps. Another possibility is the mean.
                    #time_series_data[data_name]['data'] = time_series_data[data_name]['data'].resample(f'{resolution}T').mean()
                    data = (data.resample(f"{resolution}min").ffill())
                    # We store the start date of the dataset, and the data matrix.
                    values = torch.tensor(data.values,dtype=torch.float32).detach()
                    #let's normalize it - being a thermospheric density, this strategy is different than the other time series:
                    values=scale_density(values,normalization_dict_ts)
                    time_series_data[data_name]['values']=values
                    time_series_data[data_name]['start_date']=start_date
        return time_series_data

def find_sw_from_thermo(dates,df_sw):
    # Ensure dates are in the same format
    dates = pd.to_datetime(dates).strftime("%Y-%m-%d")
    # Convert the DataFrame's index to a list of dates in 'YYYY-MM-DD' format
    dates_thermo = pd.to_datetime(df_sw['all__dates_datetime__'].values).strftime("%Y-%m-%d")
    # Create a boolean mask for the rows where the date matches any of the input dates
    mask = dates_thermo.isin(dates)
    # Use the mask to filter the DataFrame
    result_df = df_sw[mask].copy()
    # Ensure that repeated dates in the input list are reflected in the result
    result_df = result_df.loc[result_df.index.repeat(dates_thermo[mask].map(dates.value_counts().to_dict()).fillna(0).astype(int))]
    return result_df

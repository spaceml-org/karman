import torch
import pandas as pd
import numpy as np
import math
from . import nn
from .density_models import scalers_dict, normalization_dict_ts

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

def date_to_index(date, date_start, delta_seconds):
    delta_date = date - date_start
    
    #if it's not a vector, we can just return the index:
    return math.floor(delta_date.total_seconds() / delta_seconds)

def get_normalized_time_series(time_series_data_normalized,
                                start_date,
                                resolution,
                                lag,
                                date
                               ):
    now_index = date_to_index(pd.to_datetime(date), 
                                   start_date,
                                    60 * resolution)
    lagged_index = date_to_index(pd.to_datetime(date) - pd.Timedelta(minutes=lag),
                                        start_date,
                                        60 * resolution,)
    return time_series_data_normalized[lagged_index : (now_index + 1), :]

#getter for time series data:
def normalize_time_series_data(resolution,
                                data_path,
                         ):
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
                data = data.interpolate(method="pad")
                # We resample the data to the chosen resolution. We use forward fill, to fill in the gaps. Another possibility is the mean.
                #time_series_data[data_name]['data'] = time_series_data[data_name]['data'].resample(f'{resolution}T').mean()
                data = (data.resample(f"{resolution}T").ffill())
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
                    data = (data.resample(f"{resolution}T").ffill())
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

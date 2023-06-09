import datetime
import numpy as np
import math
import torch
from torch.utils.data import Dataset, Subset
import pandas as pd
from functools import lru_cache
import os
from sklearn.preprocessing import QuantileTransformer, MinMaxScaler
from tqdm import tqdm

class ThermosphericDensityDataset(Dataset):
    def __init__(
        self,
        directory='/home/jupyter/karman-project/data_directory',
        lag_minutes_omni=2*24*60,
        lag_minutes_fism2_flare_stan_bands = 12*60,
        lag_minutes_fism2_daily_stan_bands = 24*60,
        omni_resolution=60, #1 hour
        fism2_flare_stan_bands_resolution=60, #1 hour
        fism2_daily_stan_bands_resolution=24*60, #1 day
        features_to_exclude_thermo=['all__dates_datetime__', 'tudelft_thermo__satellite__',
                                    'tudelft_thermo__ground_truth_thermospheric_density__[kg/m**3]',
                                    'NRLMSISE00__thermospheric_density__[kg/m**3]',
                                    'JB08__thermospheric_density__[kg/m**3]'],
        features_to_exclude_omni=['all__dates_datetime__',
                                  'omniweb__id_for_imf_spacecraft__',
                                  'omniweb__id_for_sw_plasma_spacecraft__',
                                  'omniweb__#_of_points_in_imf_averages__',
                                  'omniweb__#_of_points_in_plasma_averages__',
                                  'omniweb__percent_of_interpolation__',
                                  'omniweb__timeshift__[s]',
                                  'omniweb__rms_timeshift__[s]',
                                  'omniweb__rms_min_variance__[s**2]',
                                  'omniweb__time_btwn_observations__[s]',
                                  'omniweb__rms_sd_b_scalar__[nT]',
                                  'omniweb__rms_sd_b_field_vector__[nT]',
                                  'omniweb__flow_pressure__[nPa]',
                                  'omniweb__electric_field__[mV/m]',
                                  'omniweb__plasma_beta__',
                                  'omniweb__alfven_mach_number__',
                                  'omniweb__magnetosonic_mach_number__',
                                  'omniweb__s/cx_gse__[Re]',
                                  'omniweb__s/cy_gse__[Re]',
                                  'omniweb__s/cz_gse__[Re]'],
        features_to_exclude_fism2_flare_stan_bands=['all__dates_datetime__'],
        features_to_exclude_fism2_daily_stan_bands=['all__dates_datetime__'],
        create_cyclical_features=True,
        min_date=pd.to_datetime('2004-02-01 00:00:00'),
        max_date=pd.to_datetime('2020-01-01 23:59:00'), # Dont change these generally. We want them fixed for all training runs
        max_altitude=np.inf,
        include_omni=False,
        include_daily_stan_bands=False,
        include_flare_stan_bands=False,
        thermo_scaler=None,
        cyclical_feature_scaler=None,
        thermospheric_density_scaler=None,
    ):
        """
        This class is used to load the dataset for the thermospheric density prediction task.
        It is a subclass of torch.utils.data.Dataset, so it can be used with PyTorch's DataLoader.
        For a description of the features supported in each dataset and their names, check the README.md file.
        
        The following input parameters are currently supported:
            - omni data, from OMNIWeb NASA website (https://omniweb.gsfc.nasa.gov/form/omni_min.html)
            - FISM2 flare stan bands, from FISM2 website (https://lasp.colorado.edu/lisird/data/fism_flare_bands/)
            - FISM2 daily stan bands, from FISM2 website (https://lasp.colorado.edu/lisird/data/fism_daily_bands/)
            - thermospheric density data, from the TU Delft website (http://thermosphere.tudelft.nl/)
            - thermospheric density data, from the NRLMSISE00 model (https://kauai.ccmc.gsfc.nasa.gov/instantrun/msis)
            - thermospheric density data, from the JB08 model (https://ccmc.gsfc.nasa.gov/modelweb/models/jb2008.php)
        The data is loaded from the following files (that must be located under the `directory` path):
            - 'data_omniweb_v1/omniweb_1min_data_2001_2022.h5'
            - 'fism2_flare_stan_bands.h5'
            - 'fism2_daily_stan_bands.h5'
            - 'data_tudelft_thermo_v1/tudelft_thermo_1min_data_2004_2020.h5'
            - 'data_nrlmsise00_thermo_v1/nrlmsise00_thermo_1min_data_2004_2020.h5'
            - 'data_jb08_thermo_v1/jb08_thermo_1min_data_2004_2020.h5'
        The loaded data supports the following time ranges:
            - omni data: 2001-01-01 00:00:00 - 2022-01-01 00:00:00
            - FISM2 flare stan bands: 2004-02-01 00:00:00 - 2020-01-01 23:59:00
            - FISM2 daily stan bands: 2004-02-01 00:00:00 - 2020-01-01 23:59:00
            - thermospheric density data, from the TU Delft website: 2004-02-01 00:00:00 - 2020-01-01 23:59:00
            - thermospheric density data, from the NRLMSISE00 model: 2004-02-01 00:00:00 - 2020-01-01 23:59:00
            - thermospheric density data, from the JB08 model: 2004-02-01 00:00:00 - 2020-01-01 23:59:00
        
        Parameters:
        ------------
            - directory (`str`): The directory where the data is stored.
            - features_to_exclude_thermo (`list`): A list of strings, each string is the name of a feature to exclude from the thermospheric density data.
            - features_to_exclude_omni (`list`): A list of strings, each string is the name of a feature to exclude from the omni data.
            - features_to_exclude_fism2_flare_stan_bands (`list`): A list of strings, each string is the name of a feature to exclude from the FISM2 flare stan bands data.
            - features_to_exclude_fism2_daily_stan_bands (`list`): A list of strings, each string is the name of a feature to exclude from the FISM2 daily stan bands data.
            - omni_resolution (`int`): The resolution of the omni data.  Default is 60 minutes.
            - fism2_flare_stan_bands_resolution (`int`): The resolution of the FISM2 flare stan bands data.  Default is 60 minutes.
            - fism2_daily_stan_bands_resolution (`int`): The resolution of the FISM2 daily stan bands data.  Default is 24*60 minutes.
            - create_cyclical_features (`bool`): Whether to create cyclical features for the time of day and the day of the year. Default is True.
            - min_date (`str`): The minimum date to load data for. The format is 'YYYY-MM-DD HH:MM:SS'. Default is '2004-02-01 00:00:00'. 
            - max_date (`str`): The maximum date to load data for. The format is 'YYYY-MM-DD HH:MM:SS'. Default is '2020-01-01 23:59:00'.
            - max_altitude (`int`): The maximum altitude to load data for. Default is inf.
            - lag_minutes_omni (`int`): The number of minutes to lag the omni data. Default is 2*24*60.
            - lag_minutes_fism2_flare_stan_bands (`int`): The number of minutes to lag the FISM2 flare stan bands data. Default is 12*60.
            - lag_minutes_fism2_daily_stan_bands (`int`): The number of minutes to lag the FISM2 daily stan bands data. Default is 24*60.
            - include_omni (`bool`): Whether to include the omni data. Default is False.
            - include_daily_stan_bands (`bool`): Whether to include the FISM2 daily stan bands data. Default is False.
            - include_flare_stan_bands (`bool`): Whether to include the FISM2 flare stan bands data. Default is False.
            - thermo_scaler (`str`): The scaler to use for the thermospheric density data. Can be 'minmax' or 'standard'. Default is None.
        Returns:
        ------------
            - `DataLoader`: A DataLoader object.
            
        """
        self.features_to_exclude_thermo = features_to_exclude_thermo
        self.features_to_exclude_fism2_daily_stan_bands=features_to_exclude_fism2_daily_stan_bands
        self.features_to_exclude_fism2_flare_stan_bands=features_to_exclude_fism2_flare_stan_bands
        self.features_to_exclude_omni=features_to_exclude_omni
        self.create_cyclical_features = create_cyclical_features
        self._directory = directory
        self.min_date = min_date
        self.max_date = max_date
        self.max_altitude = max_altitude
        self.time_series_data = {}
        self.include_omni = include_omni
        self.include_daily_stan_bands = include_daily_stan_bands
        self.include_flare_stan_bands = include_flare_stan_bands
        self.thermo_scaler = thermo_scaler
        self.thermospheric_density_scaler = thermospheric_density_scaler

        # Add time series data here.
        if self.include_omni:
            print("Loading Omni.")
            self._add_time_series_data('omni',
                                        'data_omniweb_v1/omniweb_1min_data_2001_2022.h5',
                                        lag_minutes_omni,
                                        omni_resolution,
                                        features_to_exclude_omni)

        if self.include_flare_stan_bands:
            print("Loading FISM2 Flare Stan bands.")
            self._add_time_series_data('fism2_flare_stan_bands',
                                    'fism2_flare_stan_bands.h5',
                                    lag_minutes_fism2_flare_stan_bands,
                                    fism2_flare_stan_bands_resolution,
                                    features_to_exclude_fism2_flare_stan_bands)

        if self.include_daily_stan_bands:
            print("Loading FISM2 Daily Stan bands.")
            self._add_time_series_data('fism2_daily_stan_bands',
                                    'fism2_daily_stan_bands.h5',
                                    lag_minutes_fism2_daily_stan_bands,
                                    fism2_daily_stan_bands_resolution,
                                    features_to_exclude_fism2_daily_stan_bands)

        print('Creating thermospheric density dataset')
        self.data_thermo = {}
        self.data_thermo['data'] = pd.read_hdf(os.path.join(directory,'jb08_nrlmsise_all_v1/jb08_nrlmsise_all.h5'))
        self.data_thermo['data'] = self.data_thermo['data'].sort_values('all__dates_datetime__')

        # Only include data that is between the minimum and maximum that applies to all datasets
        # These dates were calculated by looking at the available data and seeing which
        # was the minimum located in every dataset incorporating up to 100 days lag and the
        # maximum date that occurs in every dataset. Hardcoding these dates means that all
        # experiments will be run on exactly the same data, regardless of the inputted lag.
        # This means the results from all experiments will be totally comparable as they
        # are on the same observation data.
        self.data_thermo['data'] = self.data_thermo['data'][self.data_thermo['data']['all__dates_datetime__'] >= self.min_date]
        self.data_thermo['data'] = self.data_thermo['data'][self.data_thermo['data']['all__dates_datetime__'] <= self.max_date]

        #Add constraints to the data here, e.g. altitude ranges
        self.data_thermo['data'] = self.data_thermo['data'][self.data_thermo['data']['tudelft_thermo__altitude__[m]'] <= self.max_altitude]

        #Awful feature where reset_index creates a column called 'index'. drop=True gets rid of this
        self.data_thermo['data'].reset_index(inplace=True, drop=True)

        self.data_thermo['dates'] = np.array(self.data_thermo['data']['all__dates_datetime__'])
        self.data_thermo['date_start'] = self.data_thermo['dates'][0]

        # This logic creates sin and cos versions of cyclical features to avoid hard boundaries in
        # the outputted densities.
        if self.create_cyclical_features:
            print('Creating cyclical features')
            # Features to create cyclical values for
            self.cyclical_features = [
                'all__day_of_year__[d]',
                'all__seconds_in_day__[s]',
                'all__sun_right_ascension__[rad]',
                'all__sun_declination__[rad]',
                'all__sidereal_time__[rad]',
                'tudelft_thermo__longitude__[deg]',
                'tudelft_thermo__local_solar_time__[h]']
            print(features_to_exclude_thermo)
            print(self.cyclical_features)
            features_to_exclude_thermo = features_to_exclude_thermo + self.cyclical_features
            if cyclical_feature_scaler is None:
                cyclical_feature_scaler={}
                scaler_cyclical=True
            else:
                scaler_cyclical=False
            for feature in self.cyclical_features:
                # Sticking to the naming conventions here is very important.
                unit = feature.split('__')[-1][1:-1]
                if unit != 'rad':
                    if scaler_cyclical:    
                        max_ = self.data_thermo['data'][feature].max()
                        min_ = self.data_thermo['data'][feature].min()
                        cyclical_feature_scaler[feature] = {'max':max_, 'min':min_}
                    else:
                        max_=cyclical_feature_scaler[feature]['max']
                        min_=cyclical_feature_scaler[feature]['min']
                    feature_as_radian = 2*np.pi*(self.data_thermo['data'][feature].values - min_)/(max_ - min_)
                    self.data_thermo['data'][f'{feature}_sin'] = np.sin(feature_as_radian)
                    self.data_thermo['data'][f'{feature}_cos'] = np.cos(feature_as_radian)
                else:
                    self.data_thermo['data'][f'{feature}_sin'] = np.sin(self.data_thermo['data'][feature])
                    self.data_thermo['data'][f'{feature}_cos'] = np.cos(self.data_thermo['data'][feature])
        self.cyclical_feature_scaler=cyclical_feature_scaler
        print(f'Used features: {self.data_thermo["data"].drop(columns=features_to_exclude_thermo).columns}')
        self.data_thermo['data_matrix'] = self.data_thermo['data'].drop(columns=features_to_exclude_thermo).values
        self.data_thermo['data_matrix'][np.isinf(self.data_thermo['data_matrix'])]=0.
        # Why min max here and Quantile later? Because I dont want
        # to normalize cyclical features- they can stay as nice sinusoids
        if self.thermo_scaler is None:
            self.thermo_scaler = MinMaxScaler()
            self.data_thermo['data_matrix'] = self.thermo_scaler.fit_transform(self.data_thermo['data_matrix'])#.astype(np.float32)
        else:
            self.data_thermo['data_matrix'] = self.thermo_scaler.transform(self.data_thermo['data_matrix'])#.astype(np.float32)

        self.data_thermo['data_matrix']  = torch.tensor(self.data_thermo['data_matrix'] ).detach()
        self.data_thermo['scaler'] = self.thermo_scaler

        # Normalize the thermospheric density.
        self.thermospheric_density = self.data_thermo['data']['tudelft_thermo__ground_truth_thermospheric_density__[kg/m**3]'].values
        thermospheric_density_log=np.log(self.thermospheric_density*1e12)
        if self.thermospheric_density_scaler is None:
            self.thermospheric_density_scaler={}
            self.thermospheric_density_scaler['log_min']=thermospheric_density_log.min()
            self.thermospheric_density_scaler['log_max']=thermospheric_density_log.max()
        self.thermospheric_density=self.minmax_normalize(thermospheric_density_log, self.thermospheric_density_scaler['log_min'], self.thermospheric_density_scaler['log_max'])
        self.thermospheric_density = torch.tensor(self.thermospheric_density).detach()

        print("\nFinished Creating dataset.")

    def _add_time_series_data(self, data_name, data_file, lag, resolution, excluded_features):
        """
        Takes an underlying data files, stored in an hdf format, and loads and normalizes the data.
        It assumes that the data has an unbroken dates column 'all__dates_datetime__' at the same interval
        from start to finish of the data. It replaces NaNs and +/-inf by interpolating them away. It also removes outliers
        by removing data points that are 99.8% away from the mean.
        The method uses a QuantileTransformer class to scale the data. The reason is it is a very convenient way
        to scale data into a forced normal distribution
        It resamples the dataset to a chosen resolution, specified in minutes. It modifies the data inplace, by creating a new
        attribute to the dataset class, called time_series_data, which is a dictionary. 
        The motivation behind this method is to provide a common api to add any time series dataset to be attached to
        each thermospheric observation, making it easy to add new time series data into the logic.
        Parameters:
        ------------
            - data_name (`str`): The name of the dataset. This is used to store the dataset in the 'time_series_data' dictionary. 
            - data_file (`str`): The name of the hdf file containing the data. It is assumed to be in the same directory as the
                thermospheric data.
            - lag (`int`): The number of time steps to lag the data by. This is used to create a time series dataset. 
            - resolution (`int`): The resolution of the dataset in minutes.
            - excluded_features (`list`): A list of features to exclude from the dataset. This is useful if you want to exclude
                features that are not needed for the model.
            
        Returns:
        ------------
            None
        """
        # Data loading:
        self.time_series_data[data_name] = {}
        self.time_series_data[data_name]['data'] = pd.read_hdf(os.path.join(self._directory, data_file))
        # we now index the data by the datetime column, and sort it by the index. The reason is that it is then easier to resample
        self.time_series_data[data_name]['data'].index = pd.to_datetime(self.time_series_data[data_name]['data']['all__dates_datetime__'])
        self.time_series_data[data_name]['data'].sort_index(inplace=True)
        # We exclude the columns that are not needed for the model.
        self.time_series_data[data_name]['data'] = self.time_series_data[data_name]['data'].drop(columns=excluded_features, axis=1)
        # This is to remove significant outliers, such as the fism2 flare data which has 10^45 photons at one point. Regardless
        # of whther this is true or not, it severely affects the distribution.
        for column in self.time_series_data[data_name]['data'].columns:
            quantile = self.time_series_data[data_name]['data'][column].quantile(0.998)
            more_than = self.time_series_data[data_name]['data'][column] >= quantile
            self.time_series_data[data_name]['data'].loc[more_than, column] = None
        # We replace NaNs and +/-inf by interpolating them away.
        self.time_series_data[data_name]['data'] = self.time_series_data[data_name]['data'].replace([np.inf, -np.inf], None)
        self.time_series_data[data_name]['data'] = self.time_series_data[data_name]['data'].interpolate(method='pad')
        # We resample the data to the chosen resolution. We use forward fill, to fill in the gaps. Another possibility is the mean.
#        self.time_series_data[data_name]['data'] = self.time_series_data[data_name]['data'].resample(f'{resolution}T').mean()
        self.time_series_data[data_name]['data'] = self.time_series_data[data_name]['data'].resample(f'{resolution}T').ffill()
        # We store the start date of the dataset, and the data matrix.
        self.time_series_data[data_name]['date_start'] = min(self.time_series_data[data_name]['data'].index)
        self.time_series_data[data_name]['data_matrix'] = self.time_series_data[data_name]['data'].values
        # Some of the time series data is highly skewed, so much so even logging doesnt help
        # This method forces a normal distribution by mapping the quantiles to
        # a normal curve.
        scaler = QuantileTransformer(output_distribution='normal', n_quantiles=10_000)
        # We scale the data, and convert it to a torch tensor.
        self.time_series_data[data_name]['data_matrix'] = scaler.fit_transform(self.time_series_data[data_name]['data_matrix'])#.astype(np.float32)
        self.time_series_data[data_name]['data_matrix'] = torch.tensor(self.time_series_data[data_name]['data_matrix']).detach()
        self.time_series_data[data_name]['lag'] = lag
        # We also store the scaler, so that we can unscale the data later.
        self.time_series_data[data_name]['scaler'] = scaler
        self.time_series_data[data_name]['resolution'] = resolution

    def unscale_density(self, scaled_density):
        logged_density = (scaled_density * (self.thermospheric_density_scaler['log_max'] - self.thermospheric_density_scaler['log_min'])) + self.thermospheric_density_scaler['log_min'] 
        return np.exp(logged_density)/1e12

    @lru_cache(maxsize=None)
    def index_to_date(self, index, date_start, delta_seconds):
        """
        This function takes an index, the start date of the dataset, and its equally spaced time interval in seconds, and returns the date corresponding
        to the provided index. The assumptions are that: 1) the index corresponds to a date that is before the dataset end; 2) the dataset is equally
        spaced in time, and its spacing corresponds to `delta_seconds` seconds
        Parameters:
        ------------
            - index (`int`): The index to convert to a date.
            - date_start (`datetime`): The start date of the dataset.
            - delta_seconds (`int`): The equally spaced time interval in seconds.
        Returns:
        ------------
            - date (`datetime`): The date corresponding to the provided index.
        """
        date=date_start+datetime.timedelta(seconds=index*delta_seconds)
        return date

    @lru_cache(maxsize=None)
    def date_to_index(self, date, date_start, delta_seconds):
        """
        This function takes a date, the start date of the dataset, and its equally spaced intervals: delta_seconds (in seconds), and returns the
        corresponding index. Note that there are two assumptions: the first is that the date is within the end date of the dataset. And the latter,
        is that the dataset must be equally spaced, by a factor of `delta_seconds`.
        Parameters:
        ------------
            - date (`datetime`): The date to convert to an index.
            - date_start (`datetime`): The start date of the dataset.
            - delta_seconds (`int`): The equally spaced time interval in seconds.
        Returns:
        ------------
            - index (`int`): The index corresponding to the provided date.
        """
        delta_date=date-date_start
        return math.floor(delta_date.total_seconds()/delta_seconds)

    def minmax_normalize(self, values, min_, max_):
        """
        This function takes a set of values, and normalizes them between 0 and 1, using the provided min and max values.
        Parameters:
        ------------
            - values (`np.array`): The values to normalize.
            - min_ (`float`): The minimum value of the dataset.
            - max_ (`float`): The maximum value of the dataset.
        Returns:
        ------------
            - values (`np.array`): The normalized values.
        """
        values = (values - min_)/(max_ - min_)
        return values

    def __getitem__(self, index):
        date = self.data_thermo['dates'][index]
        sample = {}
        sample['instantaneous_features'] = self.data_thermo['data_matrix'][index,:].detach()
        sample['target'] = self.thermospheric_density[index].detach()

        for data_name, data in self.time_series_data.items():
            now_index = self.date_to_index(date, data['date_start'], 60*data['resolution'])
            lagged_index = self.date_to_index(date - pd.Timedelta(minutes=data['lag']), data['date_start'], 60*data['resolution'])
            sample[data_name] = data['data_matrix'][lagged_index:(now_index+1), :].detach()
        return sample

    def __getdate__(self, date):
        sample = {}
        for data_name, data in self.time_series_data.items():
            now_index = self.date_to_index(date, data['date_start'], 60*data['resolution'])
            lagged_index = self.date_to_index(date - pd.Timedelta(minutes=data['lag']), data['date_start'], 60*data['resolution'])
            sample[data_name] = data['data_matrix'][lagged_index:(now_index+1), :].detach()
        return sample

    def _set_indices(self, test_month_idx, validation_month_idx):
        """
        Works out which indices are in the training, validation and test sets.
        Previously, a file with indices was stored in the cloud. However, this way
        simply works it out based on dates, which I feel is much safer. It takes
        only a couple of minutes.
        """
        years = list(range(self.min_date.year, self.max_date.year + 1))
        months = np.array(range(1,13))

        #Take the remaining indices (0-11 inclusive) as train months.
        train_month_idx = sorted(list(set(list(range(0,12))) - set(validation_month_idx) - set(test_month_idx)))

        self.train_indices=[]
        self.val_indices=[]
        self.test_indices=[]
        date_column = self.data_thermo['data']['all__dates_datetime__']
        print('Creating training, validation and test sets.')
        for year in tqdm(years, desc=f'{len(years)} years to iterate through.'):
            year_months_train = list(np.roll(months, year)[train_month_idx])
            year_months_val = list(np.roll(months, year)[validation_month_idx])
            year_months_test = list(np.roll(months, year)[test_month_idx])
            correct_year = (date_column.dt.year.astype(int) == int(year))
            date_as_month = date_column.dt.month
            self.train_indices+=list(self.data_thermo['data'][(correct_year & date_as_month.isin(year_months_train))].index)
            self.val_indices+=list(self.data_thermo['data'][(correct_year & date_as_month.isin(year_months_val))].index)
            self.test_indices+=list(self.data_thermo['data'][(correct_year & date_as_month.isin(year_months_test))].index)
        self.train_indices = sorted(self.train_indices)
        self.val_indices = sorted(self.val_indices)
        self.test_indices = sorted(self.test_indices)

        print('Train size:', len(self.train_indices))
        print('Validation size:', len(self.val_indices))
        print('Test size:', len(self.test_indices))

    def __len__(self):
        return len(self.data_thermo['dates'])

    # These three methods return a subset of the dataset based on the calculated indices.
    def train_dataset(self):
        return Subset(self, self.train_indices)

    def validation_dataset(self):
        return Subset(self, self.val_indices)

    def test_dataset(self):
        return Subset(self, self.test_indices)
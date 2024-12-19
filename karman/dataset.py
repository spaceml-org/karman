import pandas as pd
from sklearn.preprocessing import QuantileTransformer
from torch.utils.data import Dataset, Subset
import torch
import numpy as np
import math
#import os
#from functools import lru_cache
from tqdm import tqdm
from .util import exponential_atmosphere
import pickle as pk
import datetime

class KarmanDataset(Dataset):
    def __init__(
        self,
        thermo_path,
        lag_minutes_omni=2 * 24 * 60,
        omni_resolution=60,  # 1 hour
        lag_minutes_goes=2 * 24 * 60,
        goes_resolution=60,  # 1 hour
        lag_minutes_soho=2 * 24 * 60,
        soho_resolution=60,  # 1 hour
        lag_minutes_nrlmsise00=2*24*60,
        nrlmsise00_resolution=60,  # 1 hour
        lag_minutes_sdoml_latents=2 * 24 * 60,
        sdoml_latents_resolution=12,
        features_to_exclude_thermo=[
            "all__dates_datetime__",
            "tudelft_thermo__satellite__",
            "tudelft_thermo__ground_truth_thermospheric_density__[kg/m**3]",
            "all__year__[y]",
            "NRLMSISE00__thermospheric_density__[kg/m**3]",
#            "JB08__thermospheric_density__[kg/m**3]",
        ],
        features_toexclude_omni_indices=['all__dates_datetime__',
                                         'source__gaps_flag__'],
        features_to_exclude_omni_solar_wind=['all__dates_datetime__',
                                         'source__gaps_flag__'],
        features_to_exclude_omni_magnetic_field=['all__dates_datetime__',
                                         'source__gaps_flag__'],
        features_to_exclude_goes=['all__dates_datetime__',
                                  'source__gaps_flag__'],
        features_to_exclude_soho=['all__dates_datetime__',
                                   'source__gaps_flag__'],
        features_to_exclude_nrlmsise00=['all__dates_datetime__'],
        features_to_exclude_sdoml_latents=['all__dates_datetime__'],
        min_date=pd.to_datetime("2000-07-29 00:59:47"),
        max_date=pd.to_datetime("2024-05-31 23:59:32"),
        max_altitude=np.inf,
        normalization_dict=None,
        omni_indices_path=None,#"../data/omniweb_data/merged_omni_indices.csv"
        omni_solar_wind_path=None,#"../data/omniweb_data/merged_omni_solar_wind.csv"
        omni_magnetic_field_path=None,#"../data/omniweb_data/merged_omni_magnetic_field.csv"
        goes_256nm_path=None,#"../data/goes_data/goes_256nm_sw.csv"
        goes_284nm_path=None,#"../data/goes_data/goes_284nm_sw.csv"
        goes_304nm_path=None,#"../data/goes_data/goes_304nm_sw.csv"
        goes_1175nm_path=None,#"../data/goes_data/goes_1175nm_sw.csv"
        goes_1216nm_path=None,#"../data/goes_data/goes_1216nm_sw.csv"
        goes_1335nm_path=None,#"../data/goes_data/goes_1335nm_sw.csv"
        goes_1405nm_path=None,#"../data/goes_data/goes_1405nm_sw.csv"
        soho_path=None,#"../data/soho_data/soho_data.csv"
        nrlmsise00_path=None,#"../data/nrlmsise00_data/nrlmsise00_time_series.csv"
        sdoml_latents_path=None,#"../data/sdoml_latents/sdofm_nvae_embeddings_pca_50.csv"
        torch_type=torch.float32,
        target_type="log_density",
        exclude_mask='exclude_mask.pk'
    ):
        """
        This class is used to load the dataset for the thermospheric density prediction task.
        It is a subclass of torch.utils.data.Dataset, so it can be used with PyTorch's DataLoader.
        For a description of the features supported in each dataset and their names, check the README.md file.

        The following input parameters are currently supported:
            - omni data, from OMNIWeb NASA website (https://omniweb.gsfc.nasa.gov/form/omni_min.html)
            -
        The data is loaded from the following files (that must be located under the `directory` path):
            - 'data_omniweb_v1/omniweb_1min_data_2001_2022.h5'
            - 
        The loaded data supports the following time ranges:
            - omni data: 2001-01-01 00:00:00 - 2022-01-01 00:00:00
            - 
        Parameters:
        ------------
            - thermo_path (`str`): The directory where the data is stored.
            - lag_minutes_omni (`int`): The number of minutes to lag the omni data. Default is 2*24*60.
            - omni_resolution (`int`): The resolution of the omni data.  Default is 60 minutes.
            - features_to_exclude_thermo (`list`): A list of strings, each string is the name of a feature to exclude from the thermospheric density data.
            - features_to_exclude_omni (`list`): A list of strings, each string is the name of a feature to exclude from the omni data.
            - min_date (`str`): The minimum date to load data for. The format is 'YYYY-MM-DD HH:MM:SS'. Default is '2004-02-01 00:00:00'.
            - max_date (`str`): The maximum date to load data for. The format is 'YYYY-MM-DD HH:MM:SS'. Default is '2020-01-01 23:59:00'.
            - max_altitude (`int`): The maximum altitude to load data for. Default is inf.
            - omni_path (`str`): The path to the omni data file. Default is None.
            - normalization_dict (`dict`): dictionary that contains the min/max to be used for normalizaton. If `None`, they will be computed from the underlying data.
            - torch_type (`torch.dtype`): torch data type to use. Default is torch.float32.
            - target_type (`str`): The target to use for the thermospheric density data. Can be 'log_density', 'log_exp_residual'. Default is 'log_density'.
            - exclude_mask (`str`): The path to the exclude mask file. Default is 'exclude_mask.pk'. If None it is ignored,
                                    if instead it's passed but does not exist, it's computed and then saved
        """
        #quick check:
        if torch_type not in [torch.float32, torch.float64]:
            raise ValueError(f'torch_type must be either torch.float32 or torch.float64, instead {torch_type} provided.')
        self.torch_type=torch_type
        
        self.features_to_exclude_thermo = features_to_exclude_thermo
        self.features_to_exclude_omni_indices = features_toexclude_omni_indices
        self.features_to_exclude_omni_solar_wind = features_to_exclude_omni_solar_wind
        self.features_to_exclude_omni_magnetic_field = features_to_exclude_omni_magnetic_field
        self.features_to_exclude_goes = features_to_exclude_goes
        self.features_to_exclude_soho = features_to_exclude_soho
        self.features_to_exclude_sdoml_latents = features_to_exclude_sdoml_latents
        self.features_to_exclude_nrlmsise00 = features_to_exclude_nrlmsise00

        self.min_date = min_date
        self.max_date = max_date
        self.max_altitude = max_altitude
        self.time_series_data = {}
        
        self.exclude_mask = exclude_mask
        self.target_type = target_type
        
        self.thermo_path=thermo_path
        self.omni_indices_path=omni_indices_path
        self.omni_solar_wind_path=omni_solar_wind_path
        self.omni_magnetic_field_path=omni_magnetic_field_path
        self.goes_256nm_path=goes_256nm_path
        self.goes_284nm_path=goes_284nm_path
        self.goes_304nm_path=goes_304nm_path
        self.goes_1175nm_path=goes_1175nm_path
        self.goes_1216nm_path=goes_1216nm_path
        self.goes_1335nm_path=goes_1335nm_path
        self.goes_1405nm_path=goes_1405nm_path
        self.soho_path=soho_path
        self.nrlmsise00_path=nrlmsise00_path

        # Add time series data here.
        if self.omni_indices_path is not None:
            print("Loading Omni indices.")
            self._add_time_series_data(
                "omni_indices",
                self.omni_indices_path,
                lag_minutes_omni,
                omni_resolution,
                self.features_to_exclude_omni_indices,
            )
        if self.omni_solar_wind_path is not None:
            print("Loading Omni Solar Wind.")
            self._add_time_series_data(
                "omni_solar_wind",
                self.omni_solar_wind_path,
                lag_minutes_omni,
                omni_resolution,
                self.features_to_exclude_omni_solar_wind,
            )
        if self.omni_magnetic_field_path is not None:
            print("Loading Omni Magnetic Field.")
            self._add_time_series_data(
                "omni_magnetic_field",
                self.omni_magnetic_field_path,
                lag_minutes_omni,
                omni_resolution,
                self.features_to_exclude_omni_magnetic_field,
            )
        if goes_256nm_path is not None:
            print("Loading GOES 256 nm.")
            self._add_time_series_data(
                "goes_256nm",
                goes_256nm_path,
                lag_minutes_goes,
                goes_resolution,
                self.features_to_exclude_goes,
            )
        if goes_284nm_path is not None:
            print("Loading GOES 284 nm.")
            self._add_time_series_data(
                "goes_284nm",
                goes_284nm_path,
                lag_minutes_goes,
                goes_resolution,
                self.features_to_exclude_goes,
            )
        if goes_304nm_path is not None:
            print("Loading GOES 304 nm.")
            self._add_time_series_data(
                "goes_304nm",
                goes_304nm_path,
                lag_minutes_goes,
                goes_resolution,
                self.features_to_exclude_goes,
            )
        if goes_1175nm_path is not None:
            print("Loading GOES 1175 nm.")
            self._add_time_series_data(
                "goes_1175nm",
                goes_1175nm_path,
                lag_minutes_goes,
                goes_resolution,
                self.features_to_exclude_goes,
            )
        if goes_1216nm_path is not None:
            print("Loading GOES 1216 nm.")
            self._add_time_series_data(
                "goes_1216nm",
                goes_1216nm_path,
                lag_minutes_goes,
                goes_resolution,
                self.features_to_exclude_goes,
            )
        if goes_1335nm_path is not None:
            print("Loading GOES 1335 nm.")
            self._add_time_series_data(
                "goes_1335nm",
                goes_1335nm_path,
                lag_minutes_goes,
                goes_resolution,
                self.features_to_exclude_goes,
            )
        if goes_1405nm_path is not None:
            print("Loading GOES 1405 nm.")
            self._add_time_series_data(
                "goes_1405nm",
                goes_1405nm_path,
                lag_minutes_goes,
                goes_resolution,
                self.features_to_exclude_goes,
            )
        if soho_path is not None:
            print("Loading SOHO.")
            self._add_time_series_data(
                "soho",
                soho_path,
                lag_minutes_soho,
                soho_resolution,
                self.features_to_exclude_soho,
            )
        if sdoml_latents_path is not None:
            print("Loading SDO-FM Latents.")
            self._add_time_series_data(
                "sdoml_latents",
                sdoml_latents_path,
                lag_minutes_sdoml_latents,
                sdoml_latents_resolution,
                self.features_to_exclude_sdoml_latents,
            )
        print("Creating thermospheric density dataset")
        self.data_thermo = {}
        self.data_thermo["data"] = pd.read_csv(self.thermo_path)
        
        self.data_thermo["data"]["all__dates_datetime__"]=pd.to_datetime(self.data_thermo["data"]['all__dates_datetime__'].values)
        self.data_thermo["data"] = self.data_thermo["data"].sort_values("all__dates_datetime__")

        if self.exclude_mask is not None:    
            print("Removing from the data errors in mean absolute percentage error 200% or more in the density (between nrlmsise00 and ground truth)")
            try:
                print('loading it from file')
                with open(self.exclude_mask,'rb') as f:
                    mask=pk.load(f)
                assert len(mask)==len(self.data_thermo["data"])
            except:
                print(f'passed mask in {self.exclude_mask} does not exist or not valid, computing it')
                dates=pd.to_datetime(self.data_thermo["data"]['all__dates_datetime__'].values)
                nrlmsise00=self.data_thermo["data"]['NRLMSISE00__thermospheric_density__[kg/m**3]'].values
                ground_truth=self.data_thermo["data"]['tudelft_thermo__ground_truth_thermospheric_density__[kg/m**3]'].values
                ape=(np.abs((nrlmsise00 - ground_truth) / (nrlmsise00))) * 100#+ground_truth
                index=ape>200.
                index2=np.array([el._value/1e9 for el in dates[index].diff()])>60
                val=dates[index][index2]
                date_ranges=[]
                for el in val:
                    date_ranges.append([el-datetime.timedelta(hours=3.5),el+datetime.timedelta(hours=3.5)])
                mask = np.zeros(len(dates), dtype=bool)
                for start, end in tqdm(date_ranges):
                    # Update the mask for dates within the current range
                    mask_ = (dates >= start) & (dates <= end)
                    mask |= mask_
                print(f'saving it to pickle file: {self.exclude_mask}')
                with open(self.exclude_mask,'wb') as f:
                    pk.dump(mask,f)
            self.data_thermo["data"] = self.data_thermo["data"][~mask]
        self.data_thermo["data"].reset_index(inplace=True, drop=True)

        #indices_to_remove = []
        #for _, r in data_dates.iterrows():
        #    indices_to_remove += list(r["indices"])
        #indices_to_remove = np.unique(indices_to_remove)
        #self.data_thermo["data"] = self.data_thermo["data"].drop(
        #    index=indices_to_remove
        #)
        #self.data_thermo["data"] = self.data_thermo["data"].sort_values(
        #    "all__dates_datetime__"
        #)

        # Only include data that is between the minimum and maximum that applies to all datasets
        # These dates were calculated by looking at the available data and seeing which
        # was the minimum located in every dataset incorporating up to 100 days lag and the
        # maximum date that occurs in every dataset. Hardcoding these dates means that all
        # experiments will be run on exactly the same data, regardless of the inputted lag.
        # This means the results from all experiments will be totally comparable as they
        # are on the same observation data.
        self.data_thermo["data"] = self.data_thermo["data"][self.data_thermo["data"]["all__dates_datetime__"] >= self.min_date]
        self.data_thermo["data"] = self.data_thermo["data"][self.data_thermo["data"]["all__dates_datetime__"] <= self.max_date]

        # Add constraints to the data here, e.g. altitude ranges
        self.data_thermo["data"] = self.data_thermo["data"][self.data_thermo["data"]["tudelft_thermo__altitude__[m]"]<= self.max_altitude]

        # Awful feature where reset_index creates a column called 'index'. drop=True gets rid of this
        self.data_thermo["data"].reset_index(inplace=True, drop=True)

        self.data_thermo["dates"] = np.array(self.data_thermo["data"]["all__dates_datetime__"])
        self.dates_str=pd.to_datetime(self.data_thermo["data"]["all__dates_datetime__"].values).strftime('%Y-%m-%d %H:%M:%S.%f')
        self.data_thermo["date_start"] = self.data_thermo["dates"][0]
        
        #cyclical_features are: -'all__day_of_year__[d]', 'all__seconds_in_day__[s]', 'tudelft_thermo__longitude__[deg]']
        if normalization_dict is None:
            normalization_dict={}
            self.normalization_dict=normalization_dict
        else:
            self.normalization_dict=normalization_dict
        #first we store the thermospheric density min/max:
        
        feature='tudelft_thermo__longitude__[deg]'
        lon_rad=np.deg2rad(self.data_thermo["data"][feature].values)
        self.data_thermo["data"][f"{feature}_sin"] = np.sin(lon_rad)
        self.data_thermo["data"][f"{feature}_cos"] = np.cos(lon_rad)
        
        for feature in ['all__day_of_year__[d]','all__seconds_in_day__[s]']:
            if feature not in self.normalization_dict:                    
                max_=self.data_thermo["data"][feature].max()
                min_=self.data_thermo["data"][feature].min()
                self.normalization_dict[feature] = {"max": max_, "min": min_}
            feature_as_radian = (2* np.pi * (self.data_thermo["data"][feature].values-self.normalization_dict[feature]["min"])/ (self.normalization_dict[feature]["max"] - self.normalization_dict[feature]["min"]))
            self.data_thermo["data"][f"{feature}_sin"] = np.sin(feature_as_radian)
            self.data_thermo["data"][f"{feature}_cos"] = np.cos(feature_as_radian)
        
        self.features_to_exclude_thermo+=['tudelft_thermo__longitude__[deg]','all__day_of_year__[d]','all__seconds_in_day__[s]']
        print(f'Used features: {self.data_thermo["data"].drop(columns=self.features_to_exclude_thermo).columns}')
        
        self.data_thermo['data_matrix'] = self.data_thermo['data'].drop(columns=self.features_to_exclude_thermo)
        print(self.data_thermo['data_matrix'].columns)
        #let's normalize the rest:
        for feature in self.data_thermo["data_matrix"].columns:
            if "_sin" not in feature and "_cos" not in feature:
                print(f"feature is: {feature}")
                val=self.data_thermo["data_matrix"][feature].values
                if feature not in self.normalization_dict:                    
                    max_=val.max()
                    min_=val.min()
                    self.normalization_dict[feature] = {"max": max_, "min": min_}
                self.data_thermo["data_matrix"][feature]=2*(val - self.normalization_dict[feature]["min"])/(self.normalization_dict[feature]["max"] - self.normalization_dict[feature]["min"])-1
        self.column_names_instantaneous_features = self.data_thermo["data_matrix"].columns
        self.data_thermo["data_matrix"]=self.data_thermo["data_matrix"].values

        self.data_thermo['data_matrix'][np.isinf(self.data_thermo['data_matrix'])]=0.

        self.data_thermo['data_matrix']  = torch.tensor(self.data_thermo['data_matrix'],dtype=torch_type).detach()
        
        #store Ap average:
        self.ap_average = self.data_thermo['data']['celestrack__ap_average__'].values
        #classify the storm type (from G0 to G5):
        ap_values_bins = np.array([0, 39, 67, 111, 179, 399, 400])
        ap_values_classification={0:'G0', 
                                  1:'G1', 
                                  2:'G2', 
                                  3:'G3', 
                                  4:'G4', 
                                  5:'G5'}
        storm_classification = np.digitize(self.ap_average, ap_values_bins)-1
        self.storm_classification = [ap_values_classification[v] for v in storm_classification]
        self.ap_average = torch.tensor(self.ap_average,dtype=self.torch_type).detach()
        #classify altitude bins:
        altitude_bins_classification=  {0: '0-200 km',
                                        1:'200-250 km',
                                        2:'250-300 km',
                                        3:'300-350 km',
                                        4:'350-400 km',
                                        5:'400-450 km',
                                        6:'450-500 km',
                                        7:'500-550 km',
                                        8:'550-600 km'}
        altitude_bins = np.array([0, 200, 250, 300, 350, 400, 450, 500, 550, 600])*1e3
        altitude_classification = np.digitize(self.data_thermo['data']['tudelft_thermo__altitude__[m]'].values, altitude_bins)-1
        self.altitude_classification = [altitude_bins_classification[v] for v in altitude_classification]
        #classify solar activity levels according to F10.7 values:
        solar_activity_bins_classification={0:'F10.7: 0-70 (low)',
                                            1:'F10.7: 70-150 (moderate)',
                                            2:'F10.7: 150-200 (moderate-high)',
                                            3:'F10.7: 200 (high)'}
        solar_activity_bins = np.array([0, 70, 150, 200, 1000])
        solar_activity_classification = np.digitize(self.data_thermo['data']['space_environment_technologies__f107_obs__'].values, solar_activity_bins)-1
        self.solar_activity_classification = [solar_activity_bins_classification[v] for v in solar_activity_classification]

        
        #we store some useful baselines:
        #both the piecewise exponential model for the atmospheric density:
        self.exponential_atmosphere = exponential_atmosphere(torch.tensor(self.data_thermo["data"]["tudelft_thermo__altitude__[m]"].values,dtype=self.torch_type) / 1e3).detach()
        #as well as NRLMSISE-00 thermospheric density values
        self.nrlmsise00 = torch.tensor(self.data_thermo["data"]["NRLMSISE00__thermospheric_density__[kg/m**3]"].values,dtype=self.torch_type).detach()

        # ground truth thermospheric density
        self.thermospheric_density = torch.tensor(self.data_thermo["data"]["tudelft_thermo__ground_truth_thermospheric_density__[kg/m**3]"].values,dtype=self.torch_type).detach()
        #self.ref_density=torch.tensor(self.ref_density,dtype=self.torch_dtype).detach()

        if self.target_type not in normalization_dict:
            max_=torch.log10(self.thermospheric_density).max()
            min_=torch.log10(self.thermospheric_density).min()
            self.normalization_dict[self.target_type]={"max": max_, "min": min_}
            #elif self.target_type == "log_density_mean_std":
            #    thermospheric_density_log = np.log(self.thermospheric_density)
            #    self.ref_density = self.thermospheric_density
            #    if self.thermospheric_density_scaler is None:
            #        self.thermospheric_density_scaler = {}
            #        self.thermospheric_density_scaler[
            #            "log_mean"
            #        ] = thermospheric_density_log.mean()
            #        self.thermospheric_density_scaler[
            #            "log_std"
            #        ] = thermospheric_density_log.std()
            #    self.targets = (
            #        thermospheric_density_log
            #        - self.thermospheric_density_scaler["log_mean"]
            #    ) / self.thermospheric_density_scaler["log_std"]        
            #now the final bits..
        self.targets=self.scale_density(self.thermospheric_density)
        if nrlmsise00_path is not None:
            print("Loading NRLMSISE00.")
            self._add_time_series_data(
                "msise",
                nrlmsise00_path,
                lag_minutes_nrlmsise00,
                nrlmsise00_resolution,
                self.features_to_exclude_nrlmsise00,
            )
        # get input data shape
        self.input_dim = self.data_thermo["data_matrix"].shape[1]

        print("\nFinished Creating dataset.")

    def _add_time_series_data(
        self, data_name, data_path, lag, resolution, excluded_features
    ):
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
        if data_name in ["omni_indices", "omni_solar_wind", "omni_magnetic_field","goes_256nm","goes_284nm","goes_304nm","goes_1175nm","goes_1216nm","goes_1335nm","goes_1405nm","soho","sdoml_latents"]:
            self.time_series_data[data_name]["data"] = pd.read_csv(data_path)
            # we now index the data by the datetime column, and sort it by the index. The reason is that it is then easier to resample
            self.time_series_data[data_name]["data"].index = pd.to_datetime(self.time_series_data[data_name]["data"]["all__dates_datetime__"])
            self.time_series_data[data_name]["data"].sort_index(inplace=True)
            # We exclude the columns that are not needed for the model.
            self.time_series_data[data_name]["data"] = self.time_series_data[data_name]["data"].drop(columns=excluded_features, axis=1)
            # This is to remove significant outliers, such as the fism2 flare data which has 10^45 photons at one point. Regardless
            # of whther this is true or not, it severely affects the distribution.
            for column in self.time_series_data[data_name]["data"].columns:
                quantile = self.time_series_data[data_name]["data"][column].quantile(0.998)
                more_than = self.time_series_data[data_name]["data"][column] >= quantile
                self.time_series_data[data_name]["data"].loc[more_than, column] = None
            # We replace NaNs and +/-inf by interpolating them away.
            self.time_series_data[data_name]["data"] = self.time_series_data[data_name]["data"].replace([np.inf, -np.inf], None)
            self.time_series_data[data_name]["data"] = self.time_series_data[data_name]["data"].ffill()
            # We resample the data to the chosen resolution. We use forward fill, to fill in the gaps. Another possibility is the mean.
            #self.time_series_data[data_name]['data'] = self.time_series_data[data_name]['data'].resample(f'{resolution}T').mean()
            self.time_series_data[data_name]["data"] = (self.time_series_data[data_name]["data"].resample(f"{resolution}min").ffill())
            # We store the start date of the dataset, and the data matrix.
            self.time_series_data[data_name]["date_start"] = min(self.time_series_data[data_name]["data"].index)
            self.time_series_data[data_name]["column_names"] = self.time_series_data[data_name]["data"].columns
            self.time_series_data[data_name]["data_matrix"] = self.time_series_data[data_name]["data"].values
            # Some of the time series data is highly skewed, so much so even logging doesnt help
            # This method forces a normal distribution by mapping the quantiles to
            # a normal curve.
            scaler = QuantileTransformer(output_distribution="normal", n_quantiles=10_000)
            # We scale the data, and convert it to a torch tensor.
            self.time_series_data[data_name]["data_matrix"] = scaler.fit_transform(self.time_series_data[data_name]["data_matrix"])  # .astype(np.float32)
            self.time_series_data[data_name]["data_matrix"] = torch.tensor(self.time_series_data[data_name]["data_matrix"],dtype=self.torch_type).detach()
            self.time_series_data[data_name]["lag"] = lag
            # We also store the scaler, so that we can unscale the data later.
            self.time_series_data[data_name]["scaler"] = scaler
            self.time_series_data[data_name]["resolution"] = resolution
        elif data_name in ["msise"]:
            self.time_series_data[data_name]["data"] = pd.read_csv(data_path)
            # we now index the data by the datetime column, and sort it by the index. The reason is that it is then easier to resample
            self.time_series_data[data_name]["data"].index = pd.to_datetime(self.time_series_data[data_name]["data"]["all__dates_datetime__"])
            self.time_series_data[data_name]["data"].sort_index(inplace=True)
            # We exclude the columns that are not needed for the model.
            self.time_series_data[data_name]["data"] = self.time_series_data[data_name]["data"].drop(columns=excluded_features, axis=1)
            # We resample the data to the chosen resolution. We use forward fill, to fill in the gaps. Another possibility is the mean.
            #self.time_series_data[data_name]['data'] = self.time_series_data[data_name]['data'].resample(f'{resolution}T').mean()
            self.time_series_data[data_name]["data"] = (self.time_series_data[data_name]["data"].resample(f"{resolution}min").ffill())
            # We store the start date of the dataset, and the data matrix.
            self.time_series_data[data_name]["date_start"] = min(self.time_series_data[data_name]["data"].index)
            self.time_series_data[data_name]["column_names"] = self.time_series_data[data_name]["data"].columns
            self.time_series_data[data_name]["data_matrix"] = torch.tensor(self.time_series_data[data_name]["data"].values,dtype=self.torch_type).detach()
            #let's normalize it - being a thermospheric density, this strategy is different than the other time series:
            self.time_series_data[data_name]["data_matrix"]=self.scale_density(self.time_series_data[data_name]["data_matrix"])
            self.time_series_data[data_name]["lag"] = lag
            self.time_series_data[data_name]["scaler"] = None
            self.time_series_data[data_name]["resolution"] = resolution

    def __getitem__(self, index):
        #date = self.data_thermo["dates"][index]
        sample = {}
        sample["instantaneous_features"] = self.data_thermo["data_matrix"][index, :]
        # target, piecewise exponential, nrlmsise00 and jb08 thermospheric density
        #sample["jb08"] = self.jb08[index]
        sample["nrlmsise00"] = self.nrlmsise00[index]
        sample["exponential_atmosphere"] = self.exponential_atmosphere[index]
        sample["target"] = self.targets[index]
        sample["ground_truth"] = self.thermospheric_density[index]
        #geomagnetic storm condition, in G-classes:
        #source: https://www.spaceweatherlive.com/en/help/the-ap-index.html
        #Kp             Ap         Storm Type
        #[0-5-)       [0-39)       G0 
        #[5-,6-)      [39-67)      G1
        #[6-,7-)      [67,111)     G2
        #[7-,8-)      [111,179)    G3
        #[8-,9)       [179,400)    G4
        #[9,9]        [400-400]    G5
        sample["ap_average"] = self.ap_average[index]
        sample["geomagnetic_storm_G_class"] = self.storm_classification[index]
        sample["altitude_bins"] = self.altitude_classification[index]
        sample["solar_activity_bins"] = self.solar_activity_classification[index]
        sample["date"]=self.dates_str[index]#.strftime('%Y-%m-%d %H:%M:%S.%f')
        for data_name, data in self.time_series_data.items():
            now_index = self.date_to_index(pd.to_datetime(sample["date"]), 
                                           data["date_start"],
                                           60 * data["resolution"])
            lagged_index = self.date_to_index(pd.to_datetime(sample["date"]) - pd.Timedelta(minutes=data["lag"]),
                                              data["date_start"],
                                              60 * data["resolution"],)
            sample[data_name] = data["data_matrix"][lagged_index : (now_index + 1), :]
        return sample

    def __getdate__(self, date):
        sample = {}
        for data_name, data in self.time_series_data.items():
            now_index = self.date_to_index(date, 
                                           data["date_start"], 
                                           60 * data["resolution"])
            lagged_index = self.date_to_index(date - pd.Timedelta(minutes=data["lag"]),
                                              data["date_start"],
                                              60 * data["resolution"],)
            sample[data_name] = data["data_matrix"][lagged_index : (now_index + 1), :].detach()
        return sample

    def _set_indices(self, test_month_idx, validation_month_idx, custom=None):
        """
        Works out which indices are in the training, validation and test sets.
        Previously, a file with indices was stored in the cloud. However, this way
        simply works it out based on dates, which I feel is much safer. It takes
        only a couple of minutes.

        Parameters:
        ------------
            - test_month_idx (`list`): The indices of the months to use for the test set.
            - validation_month_idx (`list`): The indices of the months to use for the validation set.
            - custom (`dict`): The dictionary with the custom validation and test months to use for each year (example: {2020: {"validation": 2, "test":0}, 2024: {...}}).
        """
        years = list(range(self.min_date.year, self.max_date.year + 1))
        months = np.array(range(1, 13))

        # Take the remaining indices (0-11 inclusive) as train months.
        train_month_idx = sorted(
            list(
                set(list(range(0, 12)))
                - set(validation_month_idx)
                - set(test_month_idx)
            )
        )

        self.train_indices = []
        self.val_indices = []
        self.test_indices = []
        date_column = self.data_thermo["data"]["all__dates_datetime__"]
        print("Creating training, validation and test sets.")
        for idx_year, year in enumerate(tqdm(years, desc=f"{len(years)} years to iterate through.")):
            #custom events:
            if custom is not None and year in custom:
                year_months_val = [custom[year]['validation']]
                year_months_test = [custom[year]['test']]
                year_months_train=sorted(list(set(list(range(0, 12)))- set(validation_month_idx)- set(test_month_idx)))
            else:  
                year_months_train = list(np.roll(months, idx_year)[train_month_idx])
                year_months_val = list(np.roll(months, idx_year)[validation_month_idx])
                year_months_test = list(np.roll(months, idx_year)[test_month_idx])
            correct_year = date_column.dt.year.astype(int) == int(year)
            date_as_month = date_column.dt.month
            self.train_indices += list(
                self.data_thermo["data"][
                    (correct_year & date_as_month.isin(year_months_train))
                ].index
            )
            self.val_indices += list(
                self.data_thermo["data"][
                    (correct_year & date_as_month.isin(year_months_val))
                ].index
            )
            self.test_indices += list(
                self.data_thermo["data"][
                    (correct_year & date_as_month.isin(year_months_test))
                ].index
            )
        self.train_indices = sorted(self.train_indices)
        self.val_indices = sorted(self.val_indices)
        self.test_indices = sorted(self.test_indices)

        print("Train size:", len(self.train_indices))
        print("Validation size:", len(self.val_indices))
        print("Test size:", len(self.test_indices))

            
    def scale_density(self,density):
        tmp = torch.log10(density)
        log_min=self.normalization_dict[self.target_type]['min']
        log_max=self.normalization_dict[self.target_type]['max']
        return 2. * (tmp - log_min) / (log_max - log_min) - 1.

    def unscale_density(self,density_scaled):
        log_min=self.normalization_dict[self.target_type]['min']
        log_max=self.normalization_dict[self.target_type]['max']
        tmp=(log_max-log_min)*(density_scaled+1)/2+log_min
        return torch.pow(10,tmp)
#        elif self.target_type == "log_density_mean_std":
#            logged_density = (
#                self.thermospheric_density_scaler["log_mean"]
#                + scaled_density * self.thermospheric_density_scaler["log_std"]
#            )
#            return np.exp(logged_density)
    
    #@lru_cache(maxsize=1048576)
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
        delta_date = date - date_start
        #if it's not a vector, we can just return the index:
        return math.floor(delta_date.total_seconds() / delta_seconds)

    def minmax_normalize(self, values, min_, max_):
        """
        This function takes a set of values, and normalizes them between -1 and 1, using the provided min and max values.
        Parameters:
        ------------
            - values (`np.array`): The values to normalize.
            - min_ (`float`): The minimum value of the dataset.
            - max_ (`float`): The maximum value of the dataset.
        Returns:
        ------------
            - values (`np.array`): The normalized values.
        """
        values = 2*(values - min_) / (max_ - min_)-1
        return values
    
    def __len__(self):
        return len(self.data_thermo["dates"])

    # These three methods return a subset of the dataset based on the calculated indices.
    def train_dataset(self):
        return Subset(self, self.train_indices)

    def validation_dataset(self):
        return Subset(self, self.val_indices)

    def test_dataset(self):
        return Subset(self, self.test_indices)

import numpy as np
import math
import torch
from functools import lru_cache
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
from tqdm import tqdm

class ThermosphericDensityDataset(Dataset):
    def __init__(
        self,
        directory='/home/jupyter/',
        normalize=True,
        lag_minutes_omni=0,
        lag_days_fism2_daily=0,
        lag_minutes_fism2_flare=0,
        wavelength_bands_to_skip=10,
        omniweb_downsampling_ratio=1,
        exclude_fism2=False,
        exclude_omni=False,
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
        features_to_exclude_fism2_daily=['fism2_daily__uncertainty__',
                                         'all__dates_datetime__'],
        features_to_exclude_fism2_flare=['fism2_flare__uncertainty__',
                                         'all__dates_datetime__']
    ):

        self._directory = directory
        self._lag_fism2_daily=round(lag_days_fism2_daily)
        self._lag_fism2_flare=round(lag_minutes_fism2_flare/10)
        self._lag_omni=round(lag_minutes_omni/omniweb_downsampling_ratio)
        self.wavelength_bands_to_skip = wavelength_bands_to_skip
        self.exclude_fism2 = exclude_fism2
        self.exclude_omni = exclude_omni
        self.fism2_resolution = 600
        self.omni_resolution = 600
        self.fism2_daily_resolution = 86400

        print("Loading Thermospheric Density Dataset:")
        self.data_thermo=pd.read_hdf(os.path.join(directory,'jb08_nrlmsise_all_v1/jb08_nrlmsise_all.h5'))
        self.data_thermo=self.data_thermo.sort_values('all__dates_datetime__')

        if not self.exclude_omni:
            print(f"Loading OMNIWeb ({omniweb_downsampling_ratio} min) Dataset:")
            if omniweb_downsampling_ratio!=1:
                _data_omni=pd.read_hdf(os.path.join(directory, 'data_omniweb_v1/omniweb_1min_data_2001_2022.h5'))
                self.data_omni=_data_omni.iloc[::omniweb_downsampling_ratio,:]
                del _data_omni
                self.data_omni.reset_index(drop=True, inplace=True)
            else:
                self.data_omni=pd.read_hdf(os.path.join(directory, 'data_omniweb_v1/omniweb_1min_data_2001_2022.h5'))
            self.dates_omni=self.data_omni['all__dates_datetime__']
            self._date_start_omni=self.dates_omni.iloc[0]
            self.data_omni.drop(features_to_exclude_omni, axis=1, inplace=True)
            self.data_omni_matrix=self.data_omni.to_numpy().astype(np.float32)
            self.data_omni_matrix[np.isinf(self.data_omni_matrix)]=0.
            #I now make sure that the starting date of the thermospheric datasets matches the one of the FISM2 flare (which is the latest available):
            self.data_thermo=self.data_thermo[(self.data_thermo['all__dates_datetime__'] >= self.dates_omni[self._lag_omni])]
            #self.data_thermo.reset_index(drop=True, inplace=True)

        if not self.exclude_fism2:
            print("Loading FISM2 Daily Irradiance Dataset:")
            self.data_fism2_daily=pd.read_hdf(os.path.join(directory, 'fism2_daily_v1/fism2_daily.h5'))
            self.dates_fism2_daily=self.data_fism2_daily['all__dates_datetime__']
            self.data_fism2_daily.drop(features_to_exclude_fism2_daily, axis=1, inplace=True)
            self._date_start_fism2_daily=self.dates_fism2_daily.iloc[0]

            print("Loading FISM2 Flare (10min) Irradiance Dataset:")
            base_path_fism2_flare=os.path.join(directory, 'fism2_flare_v1/downsampled/10min')
            files_fism2_flare=os.listdir(base_path_fism2_flare)
            files_fism2_flare=[os.path.join(base_path_fism2_flare, file) for file in files_fism2_flare]
            files_fism2_flare=sorted(files_fism2_flare)
            df_fism2_flare=pd.read_hdf(files_fism2_flare[0])
            for file in tqdm(files_fism2_flare[1:]):
                df_fism2_flare=pd.concat([df_fism2_flare,pd.read_hdf(file)])
            #there are 8 dates that slightly deviate from 10 minutes (not much, like less than 1 min), we make sure to round them:
            df_fism2_flare['all__dates_datetime__']=df_fism2_flare['all__dates_datetime__'].round("10T")
            self.data_fism2_flare=df_fism2_flare
            self.dates_fism2_flare=self.data_fism2_flare['all__dates_datetime__']
            self.data_fism2_flare.drop(features_to_exclude_fism2_flare, axis=1, inplace=True)
            self._date_start_fism2_flare=self.dates_fism2_flare.iloc[0]
            #I now move the fism2 flare data into a numpy matrix:
            self.fism2_flare_irradiance_matrix=np.stack(self.data_fism2_flare['fism2_flare__irradiance__[W/m**2/nm]'].to_numpy()).astype(np.float32)
            self.fism2_daily_irradiance_matrix=np.stack(self.data_fism2_daily['fism2_daily__irradiance__[W/m**2/nm]'].to_numpy()).astype(np.float32)
            self.fism2_daily_irradiance_matrix=self.fism2_daily_irradiance_matrix[
                :,
                range(0,self.fism2_daily_irradiance_matrix.shape[1],self.wavelength_bands_to_skip)
            ]
            self.fism2_flare_irradiance_matrix=self.fism2_flare_irradiance_matrix[
                :,
                range(0,self.fism2_flare_irradiance_matrix.shape[1],self.wavelength_bands_to_skip)
            ]
            self.fism2_flare_irradiance_matrix[np.isinf(self.flare_daily_irradiance_matrix)]=0.
            self.fism2_daily_irradiance_matrix[np.isinf(self.fism2_daily_irradiance_matrix)]=0.
            #I now make sure that the starting date of the thermospheric datasets matches the one of the FISM2 flare (which is the latest available):
            self.data_thermo=self.data_thermo[(self.data_thermo['all__dates_datetime__'] >= self.dates_fism2_flare[self._lag_fism2_flare])]
            #self.data_thermo.reset_index(drop=True, inplace=True)
        #we now store the dates:
        self.dates_thermo=self.data_thermo['all__dates_datetime__']
        self.thermospheric_density=self.data_thermo['tudelft_thermo__ground_truth_thermospheric_density__[kg/m**3]'].to_numpy().astype(np.float32)
        self.data_thermo.drop(features_to_exclude_thermo, axis=1, inplace=True)
        self.data_thermo_matrix=self.data_thermo.to_numpy().astype(np.float32)
        self.index_list=list(self.data_thermo.index)
        #Normalization:
        if normalize:
            print("\nNormalizing data:")
            print(f"\nNormalizing thermospheric data, features: {list(self.data_thermo.columns)}")
            self.thermo_mins=self.data_thermo_matrix.min(axis=0)
            self.thermo_maxs=self.data_thermo_matrix.max(axis=0)
            for i in range(len(self.thermo_mins)):
                self.data_thermo_matrix[:,i]=self.minmax_normalize(self.data_thermo_matrix[:,i], min_=self.thermo_mins[i], max_=self.thermo_maxs[i])


            if not self.exclude_omni:
                print(f"\nNormalizing omniweb data, features: {list(self.data_omni.columns)}")
                self.omni_mins=[]
                self.omni_maxs=[]
                for i in range(self.data_omni_matrix.shape[1]):
                    self.data_omni_matrix[:,i]=self.minmax_normalize(self.data_omni_matrix[:,i], min_=val.min(), max_=val.max())
                    self.omni_mins.append(val.min())
                    self.omni_maxs.append(val.max())

            if not self.exclude_fism2:
                print(f"\nNormalizing irradiance, features: {list(self.data_fism2_flare.columns)} and {list(self.data_fism2_daily.columns)}")
                self.fism2_flare_irradiance_mins=np.abs(self.fism2_flare_irradiance_matrix.min(axis=0))
                self.fism2_daily_irradiance_mins=np.abs(self.fism2_daily_irradiance_matrix.min(axis=0))
                self.fism2_flare_log_min=[]
                self.fism2_flare_log_max=[]
                self.fism2_daily_log_min=[]
                self.fism2_daily_log_max=[]
                for i in tqdm(range(self.fism2_flare_irradiance_matrix.shape[1])):
                    fism2_flare_normalized=np.log(self.fism2_flare_irradiance_matrix[:,i]+self.fism2_flare_irradiance_mins[i]+1e-16)
                    fism2_daily_normalized=np.log(self.fism2_daily_irradiance_matrix[:,i]+self.fism2_daily_irradiance_mins[i]+1e-16)
                    self.fism2_flare_log_min.append(fism2_flare_normalized.min())
                    self.fism2_flare_log_max.append(fism2_flare_normalized.max())
                    self.fism2_flare_irradiance_matrix[:,i]=self.minmax_normalize(fism2_flare_normalized, self.fism2_flare_log_min[-1], self.fism2_flare_log_max[-1])
                    self.fism2_daily_log_min.append(fism2_daily_normalized.min())
                    self.fism2_daily_log_max.append(fism2_daily_normalized.max())
                    self.fism2_daily_irradiance_matrix[:,i]=self.minmax_normalize(fism2_daily_normalized, self.fism2_daily_log_min[-1], self.fism2_daily_log_max[-1])

            print("\nNormalizing ground truth thermospheric density, feature name: "+'tudelft_thermo__ground_truth_thermospheric_density__[kg/m**3]')
            thermospheric_density_log=np.log(self.thermospheric_density*1e12)
            self.thermospheric_density_log_min=thermospheric_density_log.min()
            self.thermospheric_density_log_max=thermospheric_density_log.max()
            self.thermospheric_density=self.minmax_normalize(thermospheric_density_log, self.thermospheric_density_log_min, self.thermospheric_density_log_max)
            print("\nFinished normalizing data.")

    def unscale_density(self, scaled_density):
        logged_density = (scaled_density * (self.thermospheric_density_log_max - self.thermospheric_density_log_min)) + self.thermospheric_density_log_min
        return np.exp(logged_density)/10e12

    @lru_cache()
    def index_to_date(self, index, date_start, delta_seconds):
        """
        This function takes an index, a date_start of the dataset, and its equally spaced time interval in seconds, and returns the date corresponding
        to the provided index. The assumptions are that: 1) the index corresponds to a date that is before the dataset end; 2) the dataset is equally
        spaced in time, and its spacing corresponds to `delta_seconds` seconds
        """
        date=date_start+datetime.timedelta(seconds=index*delta_seconds)
        return date

    @lru_cache()
    def date_to_index(self, date, date_start, delta_seconds):
        """
        This function takes a date, the date_start of the dataset, and its equally spaced intervals: delta_seconds (in seconds), and returns the
        corresponding index. Note that there are two assumptions: the first is that the date is within the end date of the dataset. And the latter,
        is that the dataset must be equally spaced, by a factor of `delta_seconds`.
        """
        delta_date=date-date_start
        return math.floor(delta_date.total_seconds()/delta_seconds)

    def meanstd_normalize(self, values, mean, std):
        values = (values-mean)/(std)
        return values

    def minmax_normalize(self, values, min_, max_):
        values = (values - min_)/(max_ - min_)
        return values

    def __getitem__(self, index):
        date = self.dates_thermo[self.index_list[index]]
        sample = {}

        if not self.exclude_fism2:
            idx_fism2_flare=self.date_to_index(date, self._date_start_fism2_flare, self.fism2_resolution)
            idx_fism2_daily=self.date_to_index(date, self._date_start_fism2_daily, self.fism2_daily_resolution)
            sample['fism2_daily'] = torch.tensor(self.fism2_daily_irradiance_matrix[idx_fism2_daily-self._lag_fism2_daily:idx_fism2_daily+1])
            sample['fism2_flare'] = torch.tensor(self.fism2_flare_irradiance_matrix[idx_fism2_flare-self._lag_fism2_flare:idx_fism2_flare+1])

        if not self.exclude_omni:
            idx_omniweb=self.date_to_index(date, self._date_start_omni, self.omni_resolution)
            sample['omni'] = torch.tensor(self.data_omni_matrix[idx_omniweb-self._lag_omni:idx_omniweb+1,:])

        sample['static_features'] = torch.tensor(self.data_thermo_matrix[index,:])
        sample['target'] = torch.tensor(self.thermospheric_density[index])

        return sample

    def __len__(self):
        return len(self.data_thermo)

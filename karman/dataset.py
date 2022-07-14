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
        normalize=False,
        lag_minutes_omni=0,
        lag_days_fism2_daily=0,
        lag_minutes_fism2_flare=0,
        features_to_exclude_thermo=['all__dates_datetime__', 'tudelft_thermo__satellite__',
                                    'tudelft_thermo__ground_truth_thermospheric_density__[kg/m**3]',
                                    'NRLMSISE00__thermospheric_density__[kg/m**3]',
                                    'JB08__thermospheric_density__[kg/m**3]'],
        features_to_exclude_omni=['all__dates_datetime__',
                                  'omniweb__#_of_points_in_imf_averages__',
                                  'omniweb__#_of_points_in_plasma_averages__',
                                  'omniweb__id_for_imf_spacecraft__',
                                  'omniweb__id_for_sw_plasma_spacecraft__',
                                  'omniweb__percent_of_interpolation__'],
        features_to_exclude_fism2_daily=['fism2_daily__uncertainty__',
                                         'all__dates_datetime__'],
        features_to_exclude_fism2_flare=['fism2_flare__uncertainty__',
                                         'all__dates_datetime__']
    ):

        self._directory = directory
        self._lag_fism2_daily=round(lag_days_fism2_daily)
        self._lag_fism2_flare=round(lag_minutes_fism2_flare/10)
        self._lag_omni=round(lag_minutes_omni)
        print("Loading Thermospheric Density Dataset:")
        self.data_thermo=pd.read_hdf(os.path.join(directory,'jb08_nrlmsise_all_v1/jb08_nrlmsise_all.h5'))
        self.data_thermo=self.data_thermo.sort_values('all__dates_datetime__')

        print("Loading OMNIWeb (1min) Dataset:")
        self.data_omni=pd.read_hdf(os.path.join(directory, 'data_omniweb_v1/omniweb_1min_data_2001_2022.h5'))
        self.dates_omni=self.data_omni['all__dates_datetime__']
        self._date_start_omni=self.dates_omni.iloc[0]
        self.data_omni.drop(features_to_exclude_omni, axis=1, inplace=True)

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
        #I now make sure that the starting date of the thermospheric datasets matches the one of the FISM2 flare (which is the latest available):
        self.data_thermo=self.data_thermo[(self.data_thermo['all__dates_datetime__'] >= self.dates_fism2_flare[self._lag_fism2_flare])]
        self.data_thermo.reset_index(drop=True, inplace=True)
        #we now store the dates:
        self.dates_thermo=self.data_thermo['all__dates_datetime__']
        self.thermospheric_density=self.data_thermo['tudelft_thermo__ground_truth_thermospheric_density__[kg/m**3]'].to_numpy().astype(np.float32)
        self.data_thermo.drop(features_to_exclude_thermo, axis=1, inplace=True)

        #I now move the fism2 flare data into a numpy matrix:
        self.fism2_flare_irradiance_matrix=np.stack(self.data_fism2_flare['fism2_flare__irradiance__[W/m**2/nm]'].to_numpy())
        self.fism2_daily_irradiance_matrix=np.stack(self.data_fism2_daily['fism2_daily__irradiance__[W/m**2/nm]'].to_numpy())

        self.data_thermo_matrix=self.data_thermo.to_numpy().astype(np.float32)
        self.data_omni_matrix=self.data_omni.to_numpy().astype(np.float32)
        #Normalization:
        if normalize:
            print("Normalizing data:")
            #TODO: implement on the fly normalization here

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

    #@lru_cache(maxsize=4096)
    def __getitem__(self, index):
        date = self.dates_thermo[index]

        idx_fism2_flare=self.date_to_index(date, self._date_start_fism2_flare, 600)
        idx_fism2_daily=self.date_to_index(date, self._date_start_fism2_daily, 86400)
        idx_omniweb=self.date_to_index(date, self._date_start_omni, 60)

        #print(date, self.dates_fism2_flare[idx_fism2_flare], self.dates_fism2_daily[idx_fism2_daily], self.dates_omni[idx_omniweb])
        inp = torch.tensor(self.data_thermo_matrix[index,:]),\
               torch.tensor(self.fism2_flare_irradiance_matrix[idx_fism2_flare-self._lag_fism2_flare:idx_fism2_flare+1]),\
               torch.tensor(self.fism2_daily_irradiance_matrix[idx_fism2_daily-self._lag_fism2_daily:idx_fism2_daily+1]),\
               torch.tensor(self.data_omni_matrix[idx_omniweb-self._lag_omni:idx_omniweb+1,:])#.to_numpy()
        out = torch.tensor(self.thermospheric_density[index])
        return inp, out

    def __len__(self):
        return len(self.data_thermo)

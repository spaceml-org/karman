import numpy as np
import os
import argparse
import datetime
import pandas as pd
import time
import sys
import glob

from io import StringIO
from pyfiglet import Figlet
from termcolor import colored

def determine_quality_flag(seconds):
    if seconds==0:
        return 0
    elif seconds < 30 * 60:
        return 1
    elif 30 * 60 <= seconds < 2 * 60 * 60:
        return 2
    elif 2 * 60 * 60 <= seconds < 24 * 60 * 60:
        return 3
    else:
        return 4

def read_and_process(file_path, column_names):
    df = pd.read_csv(file_path, sep=r'\s+', comment=";", names=column_names)
    df["all__dates_datetime__"] = pd.to_datetime(
        df["Year"].astype(str) + df["Day of Year"].astype(str), format="%Y%j"
    ) + pd.to_timedelta(df["Seconds of day"], unit="s")

    # renaming columns
    df = df.rename(
        columns={
            "CH1": "soho__irradiance_30nm__[W/m2]",
            "CH2": "soho__irradiance_25nm__[W/m2]",
        }
    )

    return df[
        [
            "all__dates_datetime__",
            "soho__irradiance_30nm__[W/m2]",
            "soho__irradiance_25nm__[W/m2]",
        ]
    ]

def process_soho_data():

    print('SOHO Data Processing')
    f = Figlet(font='5lineoblique')
    print(colored(f.renderText('KARMAN 2.0'), 'red'))
    f = Figlet(font='digital')
    print(colored(f.renderText("SOHO Data Processing"), 'blue'))
    #print(colored(f'Version {karman.__version__}\n','blue'))

    parser = argparse.ArgumentParser(description='SOHO Data Processing', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--input_dir', type=str, default='../../data/soho_data_raw', help='Input directory of raw data')
    parser.add_argument('--output_dir', type=str, default='../../data/soho_data', help='Output directory of processed data')

    opt = parser.parse_args()

    #create output directory if it does not exist
    os.makedirs(opt.output_dir, exist_ok=True)

    all_file_list = glob.glob(f"{opt.input_dir}/**/*.00", recursive=True)
    all_file_list.sort()

    column_names = [
        "Julian date",
        "Year",
        "Day of Year",
        "Seconds of day",
        "CH1",
        "CH2",
        "CH3",
        "X",
        "Y",
        "Z",
        "R (km)",
        "R (AU)",
        "First Order Flux at 1AU",
        "Central Order Flux at 1AU",
    ]
    fun=lambda file_path: read_and_process(file_path, column_names)
    soho_euv_data_df = pd.concat(map(fun, all_file_list), ignore_index=True)

    # grid df to 1 second time grain and forwardfill missing values
    if not isinstance(soho_euv_data_df.index, pd.DatetimeIndex):
        soho_euv_data_df.set_index("all__dates_datetime__", inplace=True)
    soho_euv_data_df = soho_euv_data_df.sort_index()
    soho_euv_data_df = soho_euv_data_df[~soho_euv_data_df.index.duplicated(keep='first')]
    soho_euv_data_grid_df = soho_euv_data_df.asfreq(freq="15s", method="ffill")
    # add a flag column is_original_row to the df
    soho_euv_data_grid_df["is_original_row"] = soho_euv_data_grid_df.index.isin(soho_euv_data_df.index) * 1

    print(len(soho_euv_data_grid_df))

    # find the difference between the original and forward filled rows
    original_timestamps = soho_euv_data_df.index.values
    forward_filled_timestamps = soho_euv_data_grid_df.loc[soho_euv_data_grid_df["is_original_row"] == 0].index.values
    # find the source_timestamp
    previous_timestamps = np.searchsorted(original_timestamps, forward_filled_timestamps) - 1
    previous_timestamps = original_timestamps[previous_timestamps]
    # get differnece between source_tstamp and ffill_tstamp
    seconds_diff = (forward_filled_timestamps - previous_timestamps).astype('timedelta64[s]').astype(np.int64)

    soho_euv_data_grid_df["seconds_from_source_timestamp"] = 0.0
    soho_euv_data_grid_df.loc[soho_euv_data_grid_df["is_original_row"] == 0, "seconds_from_source_timestamp"] = seconds_diff
    print(len(soho_euv_data_grid_df))

    # set quality flags
    soho_euv_data_grid_df["source__gaps_flag__"] = soho_euv_data_grid_df["seconds_from_source_timestamp"].apply(determine_quality_flag)
    #soho_euv_data_grid_df["secondary_quality_flag"] = soho_euv_data_grid_df[
    #    "seconds_from_source_timestamp"
    #].apply(determine_secondary_quality_flag)
    soho_euv_data_grid_df = soho_euv_data_grid_df.reset_index()
    soho_euv_data_grid_df.drop(columns=["is_original_row", "seconds_from_source_timestamp"], inplace=True)
    print(len(soho_euv_data_grid_df))
    #if output_dir does not exist, create it
    print('Save to csv')
    soho_euv_data_grid_df.to_csv(os.path.join(opt.output_dir,'soho_data.csv'), index=False)

if __name__ == "__main__":
    time_start = time.time()
    process_soho_data()
    print('\nTotal duration: {}'.format(time.time() - time_start))
    sys.exit(0)
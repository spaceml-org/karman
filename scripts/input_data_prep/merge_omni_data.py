import numpy as np
import os
import argparse
import datetime
import pandas as pd
import time
import sys

from pyfiglet import Figlet
from termcolor import colored
from tqdm import tqdm

import pandas as pd
import os
import numpy as np

def create_df(filenames):
    dataframes = []
    for filename in tqdm(filenames):
        df = pd.read_csv(filename)
        dataframes.append(df)
    final_dataframe = pd.concat(dataframes, ignore_index=True)
    return final_dataframe

def merge_omni():
    print('Merging Omniweb Data')
    f = Figlet(font='5lineoblique')
    print(colored(f.renderText('KARMAN 2.0'), 'red'))
    f = Figlet(font='digital')
    print(colored(f.renderText("Merging Omniweb Data"), 'blue'))
    #print(colored(f'Version {karman.__version__}\n','blue'))

    parser = argparse.ArgumentParser(description='Merging Omniweb Data', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--input_dir', type=str, default='../../data/omniweb_data', help='Input directory of Omniweb data')

    opt = parser.parse_args()

    filenames=os.listdir(opt.input_dir)
    filenames_magnetic_field=[os.path.join(opt.input_dir,f) for f in filenames if f.endswith('.csv') and f.startswith('magnetic_field')]
    filenames_solar_wind=[os.path.join(opt.input_dir,f) for f in filenames if f.endswith('.csv') and f.startswith('solar_wind')]
    filenames_indices=[os.path.join(opt.input_dir,f) for f in filenames if f.endswith('.csv') and f.startswith('indices')]
    #first check if it exists:
    if os.path.exists(os.path.join(opt.input_dir,'merged_omni_magnetic_field.csv')):
        print('merged_merged_omni_magnetic_field.csv already exists, skipping')
    else:
        final_df_magnetic_field=create_df(filenames_magnetic_field)
        final_df_magnetic_field.sort_values('all__dates_datetime__',inplace=True)
        file_path=os.path.join(opt.input_dir,'merged_omni_magnetic_field.csv')
        final_df_magnetic_field.to_csv(file_path,index=False)
        print(f' OMNI magnetic field merged dataframe created at: {file_path}')
        del final_df_magnetic_field
    #first check if it exists:
    if os.path.exists(os.path.join(opt.input_dir,'merged_omni_solar_wind.csv')):
        print('merged_omni_solar_wind.csv already exists, skipping')
    else:    
        final_df_solar_wind=create_df(filenames_solar_wind)
        final_df_solar_wind.sort_values('all__dates_datetime__',inplace=True)
        file_path=os.path.join(opt.input_dir,'merged_omni_solar_wind.csv')
        final_df_solar_wind.to_csv(file_path,index=False)
        print(f' OMNI Solar Wind merged dataframe created at: {file_path}')
        del final_df_solar_wind

    #first check if it exists:
    if os.path.exists(os.path.join(opt.input_dir,'merged_omni_indices.csv')):
        print('merged_omni_solar_wind.csv already exists, skipping')
    else:    

        final_df_indices=create_df(filenames_indices)
        final_df_indices.sort_values('all__dates_datetime__',inplace=True)
        file_path=os.path.join(opt.input_dir,'merged_omni_indices.csv')
        final_df_indices.to_csv(file_path,index=False)
    print(f' OMNI Indices merged dataframe created at: {file_path}')

if __name__ == "__main__":
    time_start = time.time()
    merge_omni()
    print('\nTotal duration: {}'.format(time.time() - time_start))
    sys.exit(0)
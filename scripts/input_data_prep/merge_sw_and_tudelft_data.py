import numpy as np
import os
import argparse
import datetime
import pandas as pd
import time
import sys


from io import StringIO
from pyfiglet import Figlet
from termcolor import colored

import pandas as pd
import os
import numpy as np


def merge_satellites_and_sw():
    print('Merging Space Weather Indices and TU Delft Thermospheric Density Data')
    f = Figlet(font='5lineoblique')
    print(colored(f.renderText('KARMAN 2.0'), 'red'))
    f = Figlet(font='digital')
    print(colored(f.renderText("Merging Space Weather Indices and TU Delft Thermospheric Density Data"), 'blue'))
    #print(colored(f'Version {karman.__version__}\n','blue'))

    parser = argparse.ArgumentParser(description='Merging Space Weather Indices and TU Delft Thermospheric Density Data', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--input_dir_tudelft', type=str, default='../../data/tudelft_data', help='Input directory of TU Delft data')
    parser.add_argument('--input_dir_sw_data', type=str, default='../../data/sw_data', help='Input directory of SW Indices data')
    parser.add_argument('--input_dir_nrlmsise00_data',type=str, default='../../data/nrlmsise00_data',help='Input directory of NRLMSISE-00 data')
    parser.add_argument('--fraction',type=str, default=0.01,help='Fraction of the data (sampled randomly) to store to file, for training, analysis, etc.')
    parser.add_argument('--output_dir', type=str, default='../../data/merged_datasets', help='Output directory of processed data')

    opt = parser.parse_args()

    #create output directory if it does not exist
    os.makedirs(opt.output_dir, exist_ok=True)

    df=pd.read_csv(os.path.join(opt.input_dir_tudelft,'satellites_data.csv'))
    dates=pd.DatetimeIndex(df['all__dates_datetime__'].values)
    df['all__dates_datetime__']=dates

    #celestrack sw data:
    celestrack_sw=pd.read_csv(os.path.join(opt.input_dir_sw_data,'celestrack_sw.csv'))
    celestrack_sw['all__dates_datetime__']=pd.DatetimeIndex(celestrack_sw['all__dates_datetime__'])
    celestrack_sw.sort_values(by='all__dates_datetime__', inplace=True)
    #now the SET ones
    set_sw=pd.read_csv(os.path.join(opt.input_dir_sw_data,'set_sw.csv'))
    set_sw['all__dates_datetime__']=pd.DatetimeIndex(set_sw['all__dates_datetime__'])
    set_sw.sort_values(by='all__dates_datetime__', inplace=True)

    sw_data = pd.merge_asof(celestrack_sw,
                                set_sw, 
                                on='all__dates_datetime__', 
                                direction='backward')

    df_merged = pd.merge_asof(df,
                            sw_data, 
                            on='all__dates_datetime__', 
                            direction='backward')
    del df, sw_data, celestrack_sw

    #now the nrlmsise00 data:
    df_nrlmsise00=pd.read_csv(os.path.join(opt.input_dir_nrlmsise00_data,'densities_msise.csv'))
    #let's add it as a column
    df_merged['NRLMSISE00__thermospheric_density__[kg/m**3]']=df_nrlmsise00.values.flatten()
    df_merged.reset_index(drop=True,inplace=True)
    df_merged.sort_values(by='all__dates_datetime__', inplace=True)
    #print('msise merged to data, let s save it to csv -> satellites_data_w_sw.csv')
    #df_merged.to_csv(os.path.join(opt.output_dir,'satellites_data_w_sw.csv'),index=False)
    #print('done')


    to_drop=[]
    for col in df_merged.columns:
        if col.startswith('Unn'):
            to_drop.append(col)
    if len(to_drop)>0:
        df_merged.drop(to_drop,axis=1,inplace=True)
        
    df_merged.replace([np.inf, -np.inf], np.nan,inplace=True)
    df_merged.dropna(axis=0,inplace=True)
    df_merged.reset_index(drop=True,inplace=True)
    df_merged.sort_values(by='all__dates_datetime__', inplace=True)
    df_merged.reset_index(drop=True,inplace=True)

    #subset of 1% of the data (taken randomly):
    print('let s now subsample the dataset to 1% (randomly), and store it:')
    sampled_values = df_merged.sample(frac=opt.fraction, random_state=1)
    sampled_values.sort_values(by='all__dates_datetime__', inplace=True)
    sampled_values.reset_index(drop=True,inplace=True)
    print('and save it')
    sampled_values.to_csv(os.path.join(opt.output_dir,'satellites_data_w_sw_2mln.csv'),index=False)

    #description csv (useful for holding statistics about the data)
    print('let s also store the summary statistics to csv')
    df_describe=df_merged.describe()
    df_describe.to_csv(os.path.join(opt.output_dir,'satellites_data_w_sw_describe.csv'),index=False)
    print('Done')

    #let's also produce a subsampled (1 day resolution) version of the data
    vals=pd.to_datetime(sampled_values['all__dates_datetime__'].values)
    new_dates=pd.to_datetime([f'{v.year}-{v.month}-{v.day}' for v in vals])
    idx_unique=np.unique(new_dates,return_index=True)[1]
    subset=sampled_values.iloc[list(idx_unique)].reset_index(drop=True)
    subset['all__dates_datetime__']=np.unique(new_dates)
    subset.to_csv(os.path.join(opt.output_dir,'satellites_data_subsampled_1d.csv'),index=False)
    
    ##now let's also print the final db:
    #df_merged.to_csv(os.path.join(opt.output_dir,'satellites_data_w_sw_no_msise.csv'),index=False)

if __name__ == "__main__":
    time_start = time.time()
    merge_satellites_and_sw()
    print('\nTotal duration: {}'.format(time.time() - time_start))
    sys.exit(0)
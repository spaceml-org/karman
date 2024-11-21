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
    parser.add_argument('--input_dir_sw_data', type=str, default='../../data/tudelft_data', help='Input directory of TU Delft data')
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
    del df

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
    #now let's also add msise density data as an extra column:
    df_merged.to_csv(os.path.join(opt.output_dir,'satellites_data_w_sw_no_msise.csv'),index=False)

if __name__ == "__main__":
    time_start = time.time()
    merge_satellites_and_sw()
    print('\nTotal duration: {}'.format(time.time() - time_start))
    sys.exit(0)
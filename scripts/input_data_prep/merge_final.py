import numpy as np
import os
import argparse
import datetime
import pandas as pd
import time
import sys

from pyfiglet import Figlet
from termcolor import colored

import pandas as pd
import os
import numpy as np

def merge_final():
    print('Preparing merged final data used for training')
    f = Figlet(font='5lineoblique')
    print(colored(f.renderText('KARMAN 2.0'), 'red'))
    f = Figlet(font='digital')
    print(colored(f.renderText("Preparing merged final data used for training"), 'blue'))
    #print(colored(f'Version {karman.__version__}\n','blue'))

    parser = argparse.ArgumentParser(description='Preparing merged final data used for training', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--input_dir_sw_and_tu_delft', type=str, default='../../data/merged_datasets', help='Input directory of SW and TU Delft merged data')
    parser.add_argument('--input_dir_nrlmsise00', type=str, default='../../data/nrlmsise00_data', help='Input directory of NRLMSISE-00 data')
    parser.add_argument('--output_dir', type=str, default='../../data/merged_datasets', help='Output directory of processed data')
    opt = parser.parse_args()

    #create output directory if it does not exist
    os.makedirs(opt.output_dir, exist_ok=True)

    print('loading data w sw and w/o msise (can take a few minutes)')
    df_merged=pd.read_csv(os.path.join(opt.input_dir_sw_and_tu_delft,'satellites_data_w_sw_no_msise.csv'))
    print('.. loaded')

    #subset of 1% of the data (taken randomly):
    print('let s now subsample the dataset to 1% (randomly), and store it:')
    sampled_values = df_merged.sample(frac=0.01, random_state=1)
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


if __name__ == "__main__":
    time_start = time.time()
    merge_final()
    print('\nTotal duration: {}'.format(time.time() - time_start))
    sys.exit(0)
import numpy as np
import os
import argparse
import datetime
import h5py
import numpy as np
import pandas as pd

import time
import sys

from io import StringIO
from tqdm import tqdm
from pyfiglet import Figlet
from termcolor import colored
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def process_sdoml_latents():
    print('SDO-FM Latents Data Processing')
    f = Figlet(font='5lineoblique')
    print(colored(f.renderText('KARMAN 2.0'), 'red'))
    f = Figlet(font='digital')
    print(colored(f.renderText("SDO-FM Latents Data Processing"), 'blue'))
    #print(colored(f'Version {karman.__version__}\n','blue'))

    parser = argparse.ArgumentParser(description='SDO-FM Latents Data Processing', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--sdoml_latents_dir', type=str, default='../../data/sdoml_latents_data_dir', help='SDO-FM latents data directory: this will be used also to store the processed data.')
    parser.add_argument('--pca_components', type=int, default=50, help='PCA components to reduce the dimensionality of the latents data.')

    opt = parser.parse_args()

    print('Reading SDOM Latents Data')
    #we start by loading the sdo-fm latents:
    with h5py.File('../../data/sdoml_latents/sdofm_nvae_embeddings.h5', 'r') as f:
        data_tmp = {key: f[key][:] for key in f.keys()}

    #create the datetime column
    datetime = pd.to_datetime({
        'year': data_tmp['year'],
        'month': data_tmp['month'],
        'day': data_tmp['day'],
        'hour': data_tmp['hour'],
        'minute': data_tmp['minute']
    })
    print("Done, now reducing the dimensionality via PCA")

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_tmp['latent'])

    pca = PCA(n_components=opt.pca_components)
    data_pca = pca.fit_transform(data_scaled)

    print("Done, now saving the PCA latents")
    #let's create the dataframe to store the PCA latents
    df={}
    df['all__dates_datetime__']=datetime
    for i in range(data_pca.shape[1]):
        df[f'sdofm__latent_{i}__']=data_pca[:,i]
    df=pd.DataFrame(df)
    df.to_csv(f'../../data/sdoml_latents/sdofm_nvae_embeddings_pca_{opt.pca_components}.csv',index=False)


if __name__ == "__main__":
    time_start = time.time()
    process_sdoml_latents()
    print('\nTotal duration: {}'.format(time.time() - time_start))
    sys.exit(0)
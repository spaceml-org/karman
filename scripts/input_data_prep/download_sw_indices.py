import wget
import os
import argparse
import time
import sys

from datetime import datetime
from pyfiglet import Figlet
from termcolor import colored
from tqdm import tqdm

def download_sw_indices():
    print('Space Weather Indices Data Downloading')
    f = Figlet(font='5lineoblique')
    print(colored(f.renderText('KARMAN 2.0'), 'red'))
    f = Figlet(font='digital')
    print(colored(f.renderText("Downloading Space-Weather Indices"), 'blue'))
    #print(colored(f'Version {karman.__version__}\n','blue'))

    parser = argparse.ArgumentParser(description='Omniweb Downloading', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--url_1', type=str, default='https://sol.spacenvironment.net/JB2008/indices/SOLFSMY.TXT', help='Website URL for JB08 Solar Proxies')
    parser.add_argument('--url_2', type=str, default='https://sol.spacenvironment.net/JB2008/indices/DTCFILE.TXT', help='Website URL for JB08 Geomagnetic Indices')
    parser.add_argument('--url_3', type=str, default='https://www.celestrak.com/SpaceData/SW-All.csv', help='Website URL for NRLMSISE-00 Space Weather Inputs')
    parser.add_argument('--sw_indices_data_dir',type=str, default='../../data/sw_indices_raw', help='Path where to store the sw indices data')
    opt = parser.parse_args()

    #if the directory does not exist, create it, recursively:
    os.makedirs(opt.sw_indices_data_dir, exist_ok=True) 

    downloaded_files = []

    for url in tqdm([opt.url_1,opt.url_2,opt.url_3]):
        print(f"downloading {url}")
        outdir = wget.download(url,opt.sw_indices_data_dir)
        print(f'downloaded at {outdir}')
        downloaded_files.append(outdir)
        
if __name__ == "__main__":
    time_start = time.time()
    download_sw_indices()
    print('\nTotal duration: {}'.format(time.time() - time_start))
    sys.exit(0)
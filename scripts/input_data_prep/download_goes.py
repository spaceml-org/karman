import wget
import os
import argparse
import time
import sys

from datetime import datetime
from pyfiglet import Figlet
from termcolor import colored
from tqdm import tqdm

def download_goes_data():
    print('GOES Data Downloading')
    f = Figlet(font='5lineoblique')
    print(colored(f.renderText('KARMAN 2.0'), 'red'))
    f = Figlet(font='digital')
    print(colored(f.renderText("Downloading Goes Data"), 'blue'))
    #print(colored(f'Version {karman.__version__}\n','blue'))

    parser = argparse.ArgumentParser(description='Soho Data Downloading', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--url_1', type=str, default='https://www.ncei.noaa.gov/data/goes-space-environment-monitor/access/science/euvs/', help='Website URL for GOES 14 and 15 Data')
    parser.add_argument('--url_2', type=str, default='https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/', help='Website URL for GOES 16, 17 and 18 Data')
    parser.add_argument('--goes_data_dir',type=str, default='../../data/goes_data_raw', help='Path where to store the Soho data')
    opt = parser.parse_args()

    os.makedirs(opt.goes_data_dir, exist_ok=True)

    y14 = [2009, 2010, 2012, 2015, 2016, 2017, 2018, 2019, 2020]
    y15 = list(range(2010,2020+1))
    y16 = list(range(2017,2024+1))
    y17 = list(range(2018,2023+1))
    y18 = list(range(2022,2024+1))

    #GOES 16
    #NOTE: if these fail, try to check what is the latest version and replace it in the string of the data url, and download that.
    for year in y16:
        dataurl = opt.url_2 + 'goes16' + '/l2/data/euvs-l2-avg1m_science/sci_euvs-l2-avg1m_g16_y' + str(year) + '_v1-0-5.nc'
        #check if already downloaded first
        if os.path.exists(os.path.join(opt.goes_data_dir,'goes16_y' + str(year) + '.nc')):
            print(f'File goes16_y{year}.nc already exists')
            continue
        else:
            download_dir = os.path.join(opt.goes_data_dir,'goes16_y') + str(year) + '.nc'
            wget_out_set = wget.download(dataurl,download_dir)
            print(f'downloaded at {wget_out_set}')
    #GOES 17
    for year in y17:
        dataurl = opt.url_2 + 'goes17' + '/l2/data/euvs-l2-avg1m_science/sci_euvs-l2-avg1m_g17_y' + str(year) + '_v1-0-5.nc'
        #check if already downloaded first
        if os.path.exists(os.path.join(opt.goes_data_dir,'goes17_y' + str(year) + '.nc')):
            print(f'File goes17_y{year}.nc already exists')
            continue
        else:    
            download_dir = os.path.join(opt.goes_data_dir,'goes17_y') + str(year) + '.nc'
            wget_out_set = wget.download(dataurl,download_dir)
            print(f'downloaded at {wget_out_set}')

    #GOES 18
    for year in y18:
        dataurl = opt.url_2 + 'goes18' + '/l2/data/euvs-l2-avg1m/dn_euvs-l2-avg1m_g18_y' + str(year) + '_v1-0-5.nc'
        print(dataurl)
        #check if already downloaded first
        if os.path.exists(os.path.join(opt.goes_data_dir,'goes18_y' + str(year) + '.nc')):
            print(f'File goes18_y{year}.nc already exists')
            continue
        else:    
            download_dir = os.path.join(opt.goes_data_dir,'goes18_y' + str(year) + '.nc')
            wget_out_set = wget.download(dataurl,download_dir)
            print(f'downloaded at {wget_out_set}')


    #GOES 14
    for year in y14:
        dataurl = opt.url_1 + 'goes14' + '/geuv-l2-avg1m/sci_geuv-l2-avg1m_g14_y' + str(year) + '_v5-0-0.nc'
        #check if already downloaded first
        if os.path.exists(os.path.join(opt.goes_data_dir,'goes14_y' + str(year) + '.nc')):
            print(f'File goes14_y{year}.nc already exists')
            continue
        else:    
            download_dir = os.path.join(opt.goes_data_dir,'goes14_y') + str(year) + '.nc'
            wget_out_set = wget.download(dataurl,download_dir)
            print(f'downloaded at {wget_out_set}')

    #GOES 15
    for year in y15:
        dataurl = opt.url_1 + 'goes15' + '/geuv-l2-avg1m/sci_geuv-l2-avg1m_g15_y' + str(year) + '_v5-0-0.nc'
        #check if already downloaded first
        if os.path.exists(os.path.join(opt.goes_data_dir,'goes15_y' + str(year) + '.nc')):
            print(f'File goes15_y{year}.nc already exists')
            continue
        else:
            download_dir = os.path.join(opt.goes_data_dir,'goes15_y') + str(year) + '.nc'
            wget_out_set = wget.download(dataurl,download_dir)
            print(f'downloaded at {wget_out_set}')


if __name__ == "__main__":
    time_start = time.time()
    download_goes_data()
    print('\nTotal duration: {}'.format(time.time() - time_start))
    sys.exit(0)
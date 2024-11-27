import wget
import os
import argparse
import time
import sys
import subprocess

from datetime import datetime
from pyfiglet import Figlet
from termcolor import colored
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

def download_year_data(output_dir, base_url, year):
    directory = f"{output_dir}/{year}"
    os.makedirs(directory, exist_ok=True)
    wget_command = [
        "wget",
        "-r",
        "-np",
        "-nH",
        "--cut-dirs=7",
        "-A",
        "*.00",
        "--reject",
        "index.html*",
        "--tries=5",  # Retry up to 5 times
        "--waitretry=2",  # Wait 2 seconds between retries
        f"{base_url}{year}/",
    ]

    subprocess.run(wget_command, cwd=directory)
    return year

def download_soho_data():
    print('SOHO Data Downloading')
    f = Figlet(font='5lineoblique')
    print(colored(f.renderText('KARMAN 2.0'), 'red'))
    f = Figlet(font='digital')
    print(colored(f.renderText("Downloading Soho Data"), 'blue'))
    #print(colored(f'Version {karman.__version__}\n','blue'))

    parser = argparse.ArgumentParser(description='Soho Data Downloading', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--year_start', type=int, default=2000, help='Start year for the download')
    parser.add_argument('--year_end', type=int, default=2024, help='Final year for the download')
    parser.add_argument('--base_url', type=str, default='https://lasp.colorado.edu/eve/data_access/eve_data/lasp_soho_sem_data/long/15_sec_avg/', help='Website URL for JB08 Geomagnetic Indices')
    parser.add_argument('--soho_data_dir',type=str, default='../../data/soho_data_raw', help='Path where to store the Soho data')
    opt = parser.parse_args()

    years=range(opt.year_start,opt.year_end+1)

    # let's parallelize the download to save time
    download_year_data_v2=lambda year: download_year_data(opt.soho_data_dir,opt.base_url,year)
    with ThreadPoolExecutor() as executor:  # in case needed, use the max_workers argument to limit the number of workers
        futures = [executor.submit(download_year_data_v2, year) for year in years]
        for future in as_completed(futures):
            year = future.result()
            print(f"Completed download for year: {year}")


if __name__ == "__main__":
    time_start = time.time()
    download_soho_data()
    print('\nTotal duration: {}'.format(time.time() - time_start))
    sys.exit(0)
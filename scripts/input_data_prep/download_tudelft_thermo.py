import wget
import os
import argparse
import time
import sys
import datetime
import zipfile
import ftplib

from pyfiglet import Figlet
from termcolor import colored
from tqdm import tqdm
from ftplib import FTP
from glob import glob
import copy

home = copy.deepcopy(os.path.dirname(os.getcwd()))

def unzip_all_files(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.zip'):
                try:
                        
                    file_path = os.path.join(root, file)
                    extract_dir = os.path.join(root, 'unzipped')
                    
                    # Check if the directory already exists
                    if os.path.exists(os.path.join(extract_dir,file[:-4])):
                        print(f"Skipping {file_path} as it is already unzipped.")
                        continue
                    
                    with zipfile.ZipFile(file_path, 'r') as zip_ref:
                        zip_ref.extractall(extract_dir)
                    print(f"Unzipped {file_path} in {extract_dir}")
                except Exception as e:
                    print(f"Error unzipping {file_path}: {e}")
                    #continue anyway:
                    continue 

def is_directory(ftp, item):
    """ Check if an item is a directory on the FTP server. """
    original_cwd = ftp.pwd()
    try:
        ftp.cwd(item)
        ftp.cwd(original_cwd)
        return True
    except Exception:
        return False

# Function to download files and directories recursively
def download_directory(ftp, path, local_dir):
    try:
        os.makedirs(local_dir, exist_ok=True)  # Ensure local directory exists
        ftp.cwd(path)
        print(f"Changed directory to {path}")
    except OSError:
        pass

    files = ftp.nlst()

    for file in files:
        local_path = os.path.join(local_dir, file)
        if is_directory(ftp, file):
            download_directory(ftp, file, local_path)
        else:
            if os.path.exists(local_path):
                print(f"File already exists: {local_path}, skipping download.                       ",end="\r")
            else:
                try:
                    with open(local_path, 'wb') as f:
                        ftp.retrbinary(f"RETR {file}", f.write)
                        print(f"Downloaded file: {file}                       ",end="\r")
                except ftplib.error_perm:
                    print(f"Cannot download: {file}                       ",end="\r")
    # Return to the parent directory on FTP and local file system
    ftp.cwd("..")
    os.chdir(os.path.dirname(local_dir))

def download_tudelft_thermospheric_density_data():
    print('TU Delft Thermospheric Density Data Downloading')
    f = Figlet(font='5lineoblique')
    print(colored(f.renderText('KARMAN 2.0'), 'red'))
    f = Figlet(font='digital')
    print(colored(f.renderText("Downloading TU Delft Thermospheric Density Data"), 'blue'))
    #print(colored(f'Version {karman.__version__}\n','blue'))
    parser = argparse.ArgumentParser(description='Omniweb Downloading', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--url_1', type=str, default='https://sol.spacenvironment.net/JB2008/indices/SOLFSMY.TXT', help='Website URL for JB08 Solar Proxies')
    parser.add_argument('--url_2', type=str, default='https://sol.spacenvironment.net/JB2008/indices/DTCFILE.TXT', help='Website URL for JB08 Geomagnetic Indices')
    parser.add_argument('--url_3', type=str, default='https://www.celestrak.com/SpaceData/SW-All.csv', help='Website URL for NRLMSISE-00 Space Weather Inputs')
    parser.add_argument('--tudelft_data_dir',type=str, default=os.path.join(home,'data/tudelft_data_raw'), help='Path where to store the sw indices data')
    parser.add_argument('--starting_directory_in_ftp',type=str, default='', help='Directory to start downloading from (empty if the objective is to download everything)')
    opt = parser.parse_args()

    ftp = FTP('thermosphere.tudelft.nl')
    ftp.login()
    #passive mode
    ftp.set_pasv(True)

    download_directory(ftp, opt.starting_directory_in_ftp, opt.tudelft_data_dir)

    #we quit the connection to the FTP server
    ftp.quit()

    #now unzip all of them:
    unzip_all_files(opt.tudelft_data_dir)

        
if __name__ == "__main__":
    time_start = time.time()
    download_tudelft_thermospheric_density_data()
    print('\nTotal duration: {}'.format(time.time() - time_start))
    sys.exit(0)
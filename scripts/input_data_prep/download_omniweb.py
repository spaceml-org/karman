import os
import subprocess
import argparse
import sys
import time

from datetime import datetime
from pyfiglet import Figlet
from termcolor import colored
from tqdm import tqdm

def download_omni():
    print('Omniweb Data Downloading')
    f = Figlet(font='5lineoblique')
    print(colored(f.renderText('KARMAN 2.0'), 'red'))
    f = Figlet(font='digital')
    print(colored(f.renderText("Downloading Omniweb"), 'blue'))
    #print(colored(f'Version {karman.__version__}\n','blue'))

    parser = argparse.ArgumentParser(description='Omniweb Downloading', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--start_date', type=str, default='20000101', help='Start date, in %Y%m%d')
    parser.add_argument('--end_date', type=str, default='20240531', help='Start date, in %Y%m%d')
    parser.add_argument('--base_url', type=str, default='https://omniweb.gsfc.nasa.gov/cgi/nx1.cgi', help='Website where the OMNIWeb data is')
    parser.add_argument('--omniweb_data_dir',type=str, default='../../data/omniweb_data_raw', help='Path where to store the downloaded data')
    opt = parser.parse_args()



    years=range(int(opt.start_date[:4]),int(opt.end_date[:4])+1)
    start_date = datetime.strptime(opt.start_date, "%Y%m%d")
    end_date = datetime.strptime(opt.end_date, "%Y%m%d")
    print(start_date.year, end_date.year)
    #we need to specify which ones we want (see: https://omniweb.gsfc.nasa.gov/html/command_line_sample.txt)
    fixed_params = "activity=retrieve&res=min&spacecraft=omni_min&vars=11&vars=12&vars=13&vars=14&vars=15&vars=16&vars=17&vars=18&vars=19&vars=20&vars=21&vars=22&vars=23&vars=24&vars=25&vars=26&vars=27&vars=28&vars=29&vars=30&vars=31&vars=32&vars=33&vars=34&vars=35&vars=36&vars=37&vars=38&vars=39&vars=40&vars=41&vars=42&scale=Linear&ymin=&ymax=&view=0&charsize=&xstyle=0&ystyle=0&symbol=0&symsize=&linestyle=solid&table=0&imagex=640&imagey=480&color=&back="
    #if the directory does not exist, create it, recursively:
    os.makedirs(opt.omniweb_data_dir, exist_ok=True)

    # Loop over each year and generate the command
    for year in tqdm(years):
        if year==end_date.year:    
            s_date = str(year)+"0101"
            e_date = str(year)+opt.end_date[4:]
        elif year==start_date.year:
            s_date = opt.start_date
            e_date = str(year)+"1231"
        else:
            s_date = str(year)+"0101"
            e_date = str(year)+"1231"
        
        # Generate the wget command
        output_file = os.path.join(opt.omniweb_data_dir,f"data_{s_date}_{e_date}.txt")
        print('check if already exists')
        if os.path.exists(output_file)==False:
            wget_command = f"sudo wget --post-data \"{fixed_params}&start_date={s_date}&end_date={e_date}\" {opt.base_url} -O {output_file}"
        
            # Execute the wget command
            print(f"Running command: {wget_command}")
            subprocess.run(wget_command, shell=True)
        else:
            print('file already exists, moving to the next one')

if __name__ == "__main__":
    time_start = time.time()
    download_omni()
    print('\nTotal duration: {}'.format(time.time() - time_start))
    sys.exit(0)
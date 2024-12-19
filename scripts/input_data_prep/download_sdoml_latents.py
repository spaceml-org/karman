import os
import argparse
import time
import sys

from pyfiglet import Figlet
from termcolor import colored

from huggingface_hub import login
from huggingface_hub import hf_hub_download

def download_sdoml_latents():
    print('SDO-FM Latents Data Downloading')
    f = Figlet(font='5lineoblique')
    print(colored(f.renderText('KARMAN 2.0'), 'red'))
    f = Figlet(font='digital')
    print(colored(f.renderText("Downloading SDO-FM Latents Data"), 'blue'))
    #print(colored(f'Version {karman.__version__}\n','blue'))

    parser = argparse.ArgumentParser(description='SDO-FM Latents Data Downloading', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--token', type=str, help='Hugging face login token')
    parser.add_argument('--sdoml_latents_data_dir',type=str, default='../../data/sdoml_latents', help='Path where to store the SDO-FM latents data')
    opt = parser.parse_args()

    login(token=opt.token)

    repo_id = "SpaceML/SDO-FM"
    filename = "sdofm_nvae_embeddings.h5"
    if not os.path.exists(opt.sdoml_latents_data_dir):
        os.makedirs(opt.sdoml_latents_data_dir)
    #download the file
    file_path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=opt.sdoml_latents_data_dir)
    print(f"Downloaded file at: {file_path}")



if __name__ == "__main__":
    time_start = time.time()
    download_sdoml_latents()
    print('\nTotal duration: {}'.format(time.time() - time_start))
    sys.exit(0)
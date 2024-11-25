import datetime
import os
import bisect
import numpy as np
import argparse
import pandas as pd
import time
import sys


from glob import glob
from tqdm import tqdm
from pyfiglet import Figlet
from termcolor import colored

LEAP_DATES = ((1981, 6, 30), (1982, 6, 30), (1983, 6, 30),
               (1985, 6, 30), (1987, 12, 31), (1989, 12, 31),
               (1990, 12, 31), (1992, 6, 30), (1993, 6, 30),
               (1994, 6, 30), (1995, 12, 31), (1997, 6, 30),
               (1998, 12, 31), (2005, 12, 31), (2008, 12, 31),
               (2012, 6, 30), (2015, 6, 30), (2016, 12, 31))

LEAP_DATES_t = tuple("{}-{:02}-{:02} 23:59:59".format(i[0], i[1], i[2]) for i in LEAP_DATES)

def GPS_to_UTC(date):
    date_str = datetime.datetime.strftime(date, '%Y-%m-%d %H:%M:%S.%f')
    n_leap = bisect.bisect(LEAP_DATES_t, date_str)
    date -= datetime.timedelta(seconds=n_leap)
    return date

def parsetime(v):
    return np.datetime64(
        datetime.datetime.strptime(v.decode(), '%Y-%m-%d %H:%M:%S.%f')
    )

def process_tudelft_data():
    print('TU Delft Thermospheric Density Data Processing')
    f = Figlet(font='5lineoblique')
    print(colored(f.renderText('KARMAN 2.0'), 'red'))
    f = Figlet(font='digital')
    print(colored(f.renderText("Processing TU Delft Thermospheric Density Data"), 'blue'))
    #print(colored(f'Version {karman.__version__}\n','blue'))

    parser = argparse.ArgumentParser(description='TU Delft Thermospheric Density Data Processing', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--input_dir', type=str, default='../../data/tudelft_data_raw', help='Input directory of raw data')
    parser.add_argument('--output_dir', type=str, default='../../data/tudelft_data', help='Output directory of processed data')

    opt = parser.parse_args()

    #create output directory if it does not exist
    os.makedirs(opt.output_dir, exist_ok=True)

    print("Processing Swarm A, B and C:\n")
    #let's start w version_01 ones:
    input_dir=os.path.join(opt.input_dir,'version_01/')
    print(input_dir)
    #SWARM a,b,c:
    swarm_a_path_to_files=sorted(glob(os.path.join(input_dir,'Swarm_data/unzipped/SA_DNS_POD_2*.txt')))
    swarm_b_path_to_files=sorted(glob(os.path.join(input_dir,'Swarm_data/unzipped/SB_DNS_POD_2*.txt')))
    swarm_c_path_to_files=sorted(glob(os.path.join(input_dir,'Swarm_data/unzipped/SC_DNS_POD_2*.txt')))
    goce_path_to_files=sorted(glob(os.path.join(input_dir, 'GOCE_data/unzipped/GO_DNS_WND_ACC_2*.txt')))

    print("SWARM-A")
    ###### SWARM A ######
    ############################################################################
    #
    # Thermosphere neutral density and crosswind data (c) by
    #
    #   Delft University of Technology
    #   Delft, The Netherlands
    #
    #   Contact: Christian Siemes
    #   Email: c.siemes@tudelft.nl
    #
    # Thermosphere neutral density and crosswind data are licensed under
    # the Creative Commons Attribution 4.0 International (CC BY 4.0) License.
    #
    # A copy of the license is provided in the same directory as
    # the data files. The license text can also be found here:
    #   https://creativecommons.org/licenses/by/4.0/
    #   https://creativecommons.org/licenses/by/4.0/legalcode
    #
    # More information on the data is provided on the webpage:
    #   http://thermosphere.tudelft.nl
    #
    ############################################################################
    #
    # Column  1:         Date (yyyy-mm-dd)
    # Column  2:         Time (hh:mm:ss.sss)
    # Column  3:         Time system: UTC or GPS (differs per mission)
    # Column  4:  f10.3  Altitude (m), GRS80
    # Column  5:  f13.8  Geodetic longitude (deg), GRS80
    # Column  6:  f13.8  Geodetic latitude (deg), GRS80
    # Column  7:   f6.3  Local solar time (h)
    # Column  8:  f13.8  Argument of latitude (deg)
    # Column  9:  e15.8  Density derived from GPS accelerations (kg/m3)
    # Fortran style format string: (a27,1x,f10.3,1X,f13.8,1X,f13.8,1X,f6.3,1X,f13.8,1X,e15.8)
    # Date/time                 alt        lon           lat           lst    arglat        dens_x         
    #                           m          deg           deg           h      deg           kg/m3    
    #check if already processed:
    if os.path.exists(os.path.join(opt.output_dir,'swarm_a_data.csv')):
        print("Already processed, skipping")
    else:
        swarm_a_dataframes=[]
        for swarm_a_path_to_file in tqdm(swarm_a_path_to_files):
            times_swarm_a = []
            data_swarm_a = []
            
            with open(swarm_a_path_to_file, 'r') as file:
                for line in file:
                    if line.startswith('#'):
                        continue
                    datetime_part, data_part = line.split(' UTC', 1)
                    times_swarm_a.append(parsetime(datetime_part.encode()))
                    data_swarm_a.append(list(map(float, data_part.split())))
            
            times_swarm_a = np.array(times_swarm_a, dtype='datetime64[us]')
            data_swarm_a = np.array(data_swarm_a)
            
            df_1 = pd.DataFrame(data=times_swarm_a, columns=['Time (UTC)'])
            df_2 = pd.DataFrame(data=data_swarm_a, columns=['alt [m]','lon [deg]','lat [deg]','lst [h]','arglat [deg]','dens_x [kg/m3]'])
            
            df = pd.concat([df_1, df_2], axis=1)
            swarm_a_dataframes.append(df)
            
        swarm_a_df = pd.concat(swarm_a_dataframes, ignore_index=True)
        swarm_a_df.to_csv(os.path.join(opt.output_dir,'swarm_a_data.csv'))
        del swarm_a_df

    print("SWARM-B")
    if os.path.exists(os.path.join(opt.output_dir,'swarm_b_data.csv')):
        print("Already processed, skipping")
    else:
        ###### SWARM B ######
        swarm_b_dataframes=[]
        for swarm_b_path_to_file in tqdm(swarm_b_path_to_files):
            times_swarm_b = []
            data_swarm_b = []
            
            with open(swarm_b_path_to_file, 'r') as file:
                for line in file:
                    if line.startswith('#'):
                        continue
                    datetime_part, data_part = line.split(' UTC', 1)
                    times_swarm_b.append(parsetime(datetime_part.encode()))
                    data_swarm_b.append(list(map(float, data_part.split())))
            
            times_swarm_b = np.array(times_swarm_b, dtype='datetime64[us]')
            data_swarm_b = np.array(data_swarm_b)
            
            df_1 = pd.DataFrame(data=times_swarm_b, columns=['Time (UTC)'])
            df_2 = pd.DataFrame(data=data_swarm_b, columns=['alt [m]','lon [deg]','lat [deg]','lst [h]','arglat [deg]','dens_x [kg/m3]'])
            
            df = pd.concat([df_1, df_2], axis=1)
            swarm_b_dataframes.append(df)
            
        swarm_b_df = pd.concat(swarm_b_dataframes, ignore_index=True)
        swarm_b_df.to_csv(os.path.join(opt.output_dir,'swarm_b_data.csv'))
        del swarm_b_df
    
    print("SWARM-C")
    if os.path.exists(os.path.join(opt.output_dir,'swarm_c_data.csv')):
        print("Already processed, skipping")
    else:
        ###### SWARM C ######
        swarm_c_dataframes=[]
        for swarm_c_path_to_file in tqdm(swarm_c_path_to_files):
            times_swarm_c = []
            data_swarm_c = []
            
            with open(swarm_c_path_to_file, 'r') as file:
                for line in file:
                    if line.startswith('#'):
                        continue
                    datetime_part, data_part = line.split(' UTC', 1)
                    times_swarm_c.append(parsetime(datetime_part.encode()))
                    data_swarm_c.append(list(map(float, data_part.split())))
            
            times_swarm_c = np.array(times_swarm_c, dtype='datetime64[us]')
            data_swarm_c = np.array(data_swarm_c)
            
            df_1 = pd.DataFrame(data=times_swarm_c, columns=['Time (UTC)'])
            df_2 = pd.DataFrame(data=data_swarm_c, columns=['alt [m]','lon [deg]','lat [deg]','lst [h]','arglat [deg]','dens_x [kg/m3]'])
            
            df = pd.concat([df_1, df_2], axis=1)
            swarm_c_dataframes.append(df)
            
        swarm_c_df = pd.concat(swarm_c_dataframes, ignore_index=True)
        swarm_c_df.to_csv(os.path.join(opt.output_dir,'swarm_c_data.csv'))
        del swarm_c_df

    print("GOCE")
    ###### GOCE ######
    ############################################################################
    #
    # Thermosphere neutral density and crosswind data (c) by
    #
    #   Delft University of Technology
    #   Delft, The Netherlands
    #
    #   Contact: Christian Siemes
    #   Email: c.siemes@tudelft.nl
    #
    # Thermosphere neutral density and crosswind data are licensed under
    # the Creative Commons Attribution 4.0 International (CC BY 4.0) License.
    #
    # A copy of the license is provided in the same directory as
    # the data files. The license text can also be found here:
    #   https://creativecommons.org/licenses/by/4.0/
    #   https://creativecommons.org/licenses/by/4.0/legalcode
    #
    # More information on the data is provided on the webpage:
    #   http://thermosphere.tudelft.nl
    #
    ############################################################################
    #
    # Column  1:         Date (yyyy-mm-dd)
    # Column  2:         Time (hh:mm:ss.sss)
    # Column  3:         Time system: UTC or GPS (differs per mission)
    # Column  4:  f10.3  Altitude (m), GRS80
    # Column  5:  f13.8  Geodetic longitude (deg), GRS80
    # Column  6:  f13.8  Geodetic latitude (deg), GRS80
    # Column  7:   f6.3  Local solar time (h)
    # Column  8:  f13.8  Argument of latitude (deg)
    # Column  9:  e15.8  Density (kg/m3)
    # Column 10:   f8.2  Crosswind speed in the zonal (Eastward) direction (m/s)
    # Column 11:   f8.2  Crosswind speed in the meridional (Northward) direction (m/s)
    # Column 12:   f8.2  Crosswind speed in the vertical (Upward) direction (m/s)
    # Column 13:  e15.8  Root sum square (RSS) density error (kg/m3)
    # Column 14:   f8.2  Root sum square (RSS) wind error (m/s)
    # Column 15:   i2    Flag 1: 0 = Data ok, 1 = Data possibly affected by outliers, missing data, or filter initialization
    # Column 16:   i2    Flag 2: 0 = Data ok, 1 = Data possibly affected by eclipse transition
    # Column 17:   i2    Flag 3: 0 = Ascending (dusk) pass, 1 = Descending (dawn) pass
    # Column 18:   i2    Flag 4: 0 = Ion thruster assumed active, 1 = ion thruster assumed inactive
    # Fortran style format string: (a27,1x,f10.3,1X,f13.8,1X,f13.8,1X,f6.3,1X,f13.8,1X,e15.8,1X,f8.2,1X,f8.2,1X,f8.2,1X,e15.8,1X,f8.2,1X,i2,1X,i2,1X,i2,1X,i2)goce_dataframes=[]
    if os.path.exists(os.path.join(opt.output_dir,'goce_data.csv')):
        print("Already processed, skipping")
    else:
        goce_dataframes=[]
        for goce_path_to_file in tqdm(goce_path_to_files):
            times_goce = []
            data_goce = []
            
            with open(goce_path_to_file, 'r') as file:
                for line in file:
                    if line.startswith('#'):
                        continue
                    datetime_part, data_part = line.split(' UTC', 1)
                    times_goce.append(parsetime(datetime_part.encode()))
                    data_goce.append(list(map(float, data_part.split())))
            
            times_goce = np.array(times_goce, dtype='datetime64[us]')
            data_goce = np.array(data_goce)
            
            df_1 = pd.DataFrame(data=times_goce, columns=['Time (UTC)'])
            df_2 = pd.DataFrame(data=data_goce, columns=['alt [m]','lon [deg]','lat [deg]','lst [h]','arglat [deg]','dens_x [kg/m3]','cr_wnd_e [m/s]','cr_wnd_n [m/s]','cr_wnd_u [m/s]','denserror [kg/m3]','winderro [m/s]','f1','f2','f3','f4'])
            
            df = pd.concat([df_1, df_2], axis=1)
            goce_dataframes.append(df)
            
        goce_df = pd.concat(goce_dataframes, ignore_index=True)
        goce_df.to_csv(os.path.join(opt.output_dir,'goce_data.csv'))
        del goce_df
        #goce_df.to_hdf('goce_data.h5', key='goce_df')

    print("Now GRACE A, B and C:\n")
    #let's start w version_01 ones:
    input_dir=os.path.join(opt.input_dir,'version_02/')
    #SWARM a,b,c:
    grace_a_path_to_files=sorted(glob(os.path.join(input_dir,'GRACE_data/unzipped/GA_DNS_ACC_*.txt')))
    grace_b_path_to_files=sorted(glob(os.path.join(input_dir,'GRACE_data/unzipped/GB_DNS_ACC_*.txt')))
    grace_c_path_to_files=sorted(glob(os.path.join(input_dir,'GRACE-FO_data/unzipped/GC_DNS_ACC_*.txt')))
    champ_path_to_files=sorted(glob(os.path.join(input_dir, 'CHAMP_data/unzipped/CH_DNS_ACC_2*.txt')))

    print("GRACE A")
    ###### GRACE A ######
    ############################################################################
    #
    # Thermosphere neutral density and crosswind data (c) by
    #
    #   Delft University of Technology
    #   Delft, The Netherlands
    #
    #   Contact: Christian Siemes
    #   Email: c.siemes@tudelft.nl
    #
    # Thermosphere neutral density and crosswind data are licensed under
    # the Creative Commons Attribution 4.0 International (CC BY 4.0) License.
    #
    # A copy of the license is provided in the same directory as
    # the data files. The license text can also be found here:
    #   https://creativecommons.org/licenses/by/4.0/
    #   https://creativecommons.org/licenses/by/4.0/legalcode
    #
    # More information on the data is provided on the webpage:
    #   http://thermosphere.tudelft.nl
    #
    ############################################################################
    #
    # Column  1:         Date (yyyy-mm-dd)
    # Column  2:         Time (hh:mm:ss.sss)
    # Column  3:         Time system: UTC or GPS (differs per mission)
    # Column  4:  f10.3  Altitude (m), GRS80
    # Column  5:   f8.3  Geodetic longitude (deg), GRS80
    # Column  6:   f7.3  Geodetic latitude (deg), GRS80
    # Column  7:   f6.3  Local solar time (h)
    # Column  8:   f7.3  Argument of latitude (deg)
    # Column  9:  e15.8  Density derived from accelerometer measurements (kg/m3)
    # Column 10:  e15.8  Running orbit average of density (kg/m3)
    # Column 11:   f4.1  Flag for density: 0 = nominal data, 1 = anomalous data (-)
    # Column 12:   f4.1  Flag for running orbit average density: 0 = nominal data, 1 = anomalous data (-)
    # Fortran style format string: (a27,1x,f10.3,1X,f8.3,1X,f7.3,1X,f6.3,1X,f7.3,1X,e15.8,1X,e15.8,1X,f4.1,1X,f4.1)
    # Date/time                 alt        lon      lat     lst    arglat  dens_x          dens_mean       flag flag
    #                           m          deg      deg     h      deg     kg/m3           kg/m3           -    -  
    if os.path.exists(os.path.join(opt.output_dir,'grace_a_data.csv')):
        print("Already processed, skipping")
    else:
        grace_a_dataframes=[]
        for grace_a_path_to_file in tqdm(grace_a_path_to_files):
            times_grace_a = []
            data_grace_a = []
            
            with open(grace_a_path_to_file, 'r') as file:
                for line in file:
                    if line.startswith('#'):
                        continue
                    datetime_part, data_part = line.split(' GPS', 1)
                    date=datetime.datetime.strptime(datetime_part, '%Y-%m-%d %H:%M:%S.%f')
                    n_leap = bisect.bisect(LEAP_DATES_t, datetime_part)
                    date -= datetime.timedelta(seconds=n_leap)
                    datetime_part = datetime.datetime.strftime(date, '%Y-%m-%d %H:%M:%S.%f')

                    times_grace_a.append(parsetime(datetime_part.encode()))
                    data_grace_a.append(list(map(float, data_part.split())))
            
            times_grace_a = np.array(times_grace_a, dtype='datetime64[us]')
            data_grace_a = np.array(data_grace_a)
            
            df_1 = pd.DataFrame(data=times_grace_a, columns=['Time (UTC)'])
            df_2 = pd.DataFrame(data=data_grace_a, columns=['alt [m]','lon [deg]','lat [deg]','lst [h]','arglat [deg]','dens_x [kg/m3]','dens_mean','f1','f2'])
            
            df = pd.concat([df_1, df_2], axis=1)
            grace_a_dataframes.append(df)
            
        grace_a_df = pd.concat(grace_a_dataframes, ignore_index=True)
        grace_a_df.to_csv(os.path.join(opt.output_dir,'grace_a_data.csv'))
        del grace_a_df
        #grace_a_df.to_hdf('grace_a_data.h5', key='grace_a_df')

    print("GRACE B")
    ###### GRACE B ######
    ############################################################################
    #
    # Thermosphere neutral density and crosswind data (c) by
    #
    #   Delft University of Technology
    #   Delft, The Netherlands
    #
    #   Contact: Christian Siemes
    #   Email: c.siemes@tudelft.nl
    #
    # Thermosphere neutral density and crosswind data are licensed under
    # the Creative Commons Attribution 4.0 International (CC BY 4.0) License.
    #
    # A copy of the license is provided in the same directory as
    # the data files. The license text can also be found here:
    #   https://creativecommons.org/licenses/by/4.0/
    #   https://creativecommons.org/licenses/by/4.0/legalcode
    #
    # More information on the data is provided on the webpage:
    #   http://thermosphere.tudelft.nl
    #
    ############################################################################
    #
    # Column  1:         Date (yyyy-mm-dd)
    # Column  2:         Time (hh:mm:ss.sss)
    # Column  3:         Time system: UTC or GPS (differs per mission)
    # Column  4:  f10.3  Altitude (m), GRS80
    # Column  5:   f8.3  Geodetic longitude (deg), GRS80
    # Column  6:   f7.3  Geodetic latitude (deg), GRS80
    # Column  7:   f6.3  Local solar time (h)
    # Column  8:   f7.3  Argument of latitude (deg)
    # Column  9:  e15.8  Density derived from accelerometer measurements (kg/m3)
    # Column 10:  e15.8  Running orbit average of density (kg/m3)
    # Column 11:   f4.1  Flag for density: 0 = nominal data, 1 = anomalous data (-)
    # Column 12:   f4.1  Flag for running orbit average density: 0 = nominal data, 1 = anomalous data (-)
    # Fortran style format string: (a27,1x,f10.3,1X,f8.3,1X,f7.3,1X,f6.3,1X,f7.3,1X,e15.8,1X,e15.8,1X,f4.1,1X,f4.1)
    # Date/time                 alt        lon      lat     lst    arglat  dens_x          dens_mean       flag flag
    #                           m          deg      deg     h      deg     kg/m3           kg/m3           -    -  
    if os.path.exists(os.path.join(opt.output_dir,'grace_b_data.csv')):
        print("Already processed, skipping")
    else:
        grace_b_dataframes=[]
        for grace_b_path_to_file in tqdm(grace_b_path_to_files):
            times_grace_b = []
            data_grace_b = []
            
            with open(grace_b_path_to_file, 'r') as file:
                for line in file:
                    if line.startswith('#'):
                        continue
                    datetime_part, data_part = line.split(' GPS', 1)
                    date=datetime.datetime.strptime(datetime_part, '%Y-%m-%d %H:%M:%S.%f')
                    n_leap = bisect.bisect(LEAP_DATES_t, datetime_part)
                    date -= datetime.timedelta(seconds=n_leap)
                    datetime_part = datetime.datetime.strftime(date, '%Y-%m-%d %H:%M:%S.%f')

                    times_grace_b.append(parsetime(datetime_part.encode()))
                    data_grace_b.append(list(map(float, data_part.split())))
            
            times_grace_b = np.array(times_grace_b, dtype='datetime64[us]')
            data_grace_b = np.array(data_grace_b)
            
            df_1 = pd.DataFrame(data=times_grace_b, columns=['Time (UTC)'])
            df_2 = pd.DataFrame(data=data_grace_b, columns=['alt [m]','lon [deg]','lat [deg]','lst [h]','arglat [deg]','dens_x [kg/m3]','dens_mean','f1','f2'])
            
            df = pd.concat([df_1, df_2], axis=1)
            grace_b_dataframes.append(df)
            
        grace_b_df = pd.concat(grace_b_dataframes, ignore_index=True)
        grace_b_df.to_csv(os.path.join(opt.output_dir,'grace_b_data.csv'))
        del grace_b_df
        #grace_b_df.to_hdf('grace_b_data.h5', key='grace_b_df')

    print("GRACE C")
    ###### GRACE C ######
    ############################################################################
    #
    # Thermosphere neutral density and crosswind data (c) by
    #
    #   Delft University of Technology
    #   Delft, The Netherlands
    #
    #   Contact: Christian Siemes
    #   Email: c.siemes@tudelft.nl
    #
    # Thermosphere neutral density and crosswind data are licensed under
    # the Creative Commons Attribution 4.0 International (CC BY 4.0) License.
    #
    # A copy of the license is provided in the same directory as
    # the data files. The license text can also be found here:
    #   https://creativecommons.org/licenses/by/4.0/
    #   https://creativecommons.org/licenses/by/4.0/legalcode
    #
    # More information on the data is provided on the webpage:
    #   http://thermosphere.tudelft.nl
    #
    ############################################################################
    #
    # Column  1:         Date (yyyy-mm-dd)
    # Column  2:         Time (hh:mm:ss.sss)
    # Column  3:         Time system: UTC or GPS (differs per mission)
    # Column  4:  f10.3  Altitude (m), GRS80
    # Column  5:   f8.3  Geodetic longitude (deg), GRS80
    # Column  6:   f7.3  Geodetic latitude (deg), GRS80
    # Column  7:   f6.3  Local solar time (h)
    # Column  8:   f7.3  Argument of latitude (deg)
    # Column  9:  e15.8  Density derived from accelerometer measurements (kg/m3)
    # Column 10:  e15.8  Running orbit average of density (kg/m3)
    # Column 11:   f4.1  Flag for density: 0 = nominal data, 1 = anomalous data (-)
    # Column 12:   f4.1  Flag for running orbit average density: 0 = nominal data, 1 = anomalous data (-)
    # Fortran style format string: (a27,1x,f10.3,1X,f8.3,1X,f7.3,1X,f6.3,1X,f7.3,1X,e15.8,1X,e15.8,1X,f4.1,1X,f4.1)
    # Date/time                 alt        lon      lat     lst    arglat  dens_x          dens_mean       flag flag
    #                           m          deg      deg     h      deg     kg/m3           kg/m3           -    -   
    if os.path.exists(os.path.join(opt.output_dir,'grace_c_data.csv')):
        print("Already processed, skipping")
    else:
        grace_c_dataframes=[]
        for grace_c_path_to_file in tqdm(grace_c_path_to_files):
            times_grace_c = []
            data_grace_c = []
            
            with open(grace_c_path_to_file, 'r') as file:
                for line in file:
                    if line.startswith('#'):
                        continue
                    datetime_part, data_part = line.split(' GPS', 1)
                    date=datetime.datetime.strptime(datetime_part, '%Y-%m-%d %H:%M:%S.%f')
                    n_leap = bisect.bisect(LEAP_DATES_t, datetime_part)
                    date -= datetime.timedelta(seconds=n_leap)
                    datetime_part = datetime.datetime.strftime(date, '%Y-%m-%d %H:%M:%S.%f')

                    times_grace_c.append(parsetime(datetime_part.encode()))
                    data_grace_c.append(list(map(float, data_part.split())))
            
            times_grace_c = np.array(times_grace_c, dtype='datetime64[us]')
            data_grace_c = np.array(data_grace_c)
            
            df_1 = pd.DataFrame(data=times_grace_c, columns=['Time (UTC)'])
            df_2 = pd.DataFrame(data=data_grace_c, columns=['alt [m]','lon [deg]','lat [deg]','lst [h]','arglat [deg]','dens_x [kg/m3]','dens_mean','f1','f2'])
            
            df = pd.concat([df_1, df_2], axis=1)
            grace_c_dataframes.append(df)
            
        grace_c_df = pd.concat(grace_c_dataframes, ignore_index=True)
        grace_c_df.to_csv(os.path.join(opt.output_dir,'grace_c_data.csv'))
        del grace_c_df
        #grace_c_df.to_hdf('grace_c_data.h5', key='grace_c_df')

    ###### CHAMP ######
    ############################################################################
    #
    # Thermosphere neutral density and crosswind data (c) by
    #
    #   Delft University of Technology
    #   Delft, The Netherlands
    #
    #   Contact: Christian Siemes
    #   Email: c.siemes@tudelft.nl
    #
    # Thermosphere neutral density and crosswind data are licensed under
    # the Creative Commons Attribution 4.0 International (CC BY 4.0) License.
    #
    # A copy of the license is provided in the same directory as
    # the data files. The license text can also be found here:
    #   https://creativecommons.org/licenses/by/4.0/
    #   https://creativecommons.org/licenses/by/4.0/legalcode
    #
    # More information on the data is provided on the webpage:
    #   http://thermosphere.tudelft.nl
    #
    ############################################################################
    #
    # Column  1:         Date (yyyy-mm-dd)
    # Column  2:         Time (hh:mm:ss.sss)
    # Column  3:         Time system
    # Column  4:  f10.3  CH_TOLEOS_V1/Orbit_Geo - Altitude (m)
    # Column  5:  f13.8  CH_TOLEOS_V1/Orbit_Geo - Geodetic longitude (deg)
    # Column  6:  f13.8  CH_TOLEOS_V1/Orbit_Geo - Geodetic latitude (deg)
    # Column  7:   f6.3  CH_TOLEOS_V1/Orbit_Geo - Local solar time (h)
    # Column  8:  f13.8  CH_TOLEOS_V1/Orbit_Geo - Argument of latitude (deg)
    # Column  9:  e15.8  CH_TOLEOS_V1/Density_Direct_Ray_Tracing - Density from the X-component of the acceleration (kg/m3)
    # Column 10:  e15.8  CH_TOLEOS_V1/Density_Direct_Ray_Tracing_Orbit_Mean - Running orbit average of density (kg/m3)
    # Column 11:   f4.1  CH_TOLEOS_V1/DensityDataFlagsFinal - Flag: 0 = nominal data, 1 = anomalous data (-)
    # Column 12:   f4.1  CH_TOLEOS_V1/DensityOrbitMeanDataFlagsFinal - Flag: 0 = nominal data, 1 = anomalous data (-)
    # Format string: (a27,1x,f10.3,1X,f13.8,1X,f13.8,1X,f6.3,1X,f13.8,1X,e15.8,1X,e15.8,1X,f4.1,1X,f4.1)
    # Date/time                 alt        lon           lat           lst    arglat        rho_x           rho_mean        Colu Colu
    #                           m          deg           deg           h      deg           kg/m3           kg/m3           -    -   
    if os.path.exists(os.path.join(opt.output_dir,'champ_data.csv')):
        print("Already processed, skipping")
    else:
        champ_dataframes=[]
        for champ_path_to_file in tqdm(champ_path_to_files):
            times_champ = []
            data_champ = []
            
            with open(champ_path_to_file, 'r') as file:
                for line in file:
                    if line.startswith('#'):
                        continue
                    datetime_part, data_part = line.split(' GPS', 1)
                    date=datetime.datetime.strptime(datetime_part, '%Y-%m-%d %H:%M:%S.%f')
                    n_leap = bisect.bisect(LEAP_DATES_t, datetime_part)
                    date -= datetime.timedelta(seconds=n_leap)
                    datetime_part = datetime.datetime.strftime(date, '%Y-%m-%d %H:%M:%S.%f')
                    times_champ.append(parsetime(datetime_part.encode()))
                    data_champ.append(list(map(float, data_part.split())))
            
            
            times_champ = np.array(times_champ, dtype='datetime64[us]')
            data_champ = np.array(data_champ)
            
            df_1 = pd.DataFrame(data=times_champ, columns=['Time (UTC)'])
            df_2 = pd.DataFrame(data=data_champ, columns=['alt [m]','lon [deg]','lat [deg]','lst [h]','arglat [deg]','dens_x [kg/m3]','rho_mean','f1','f2'])
            
            df = pd.concat([df_1, df_2], axis=1)
            champ_dataframes.append(df)
            
        champ_df = pd.concat(champ_dataframes, ignore_index=True)
        #champ_df.to_hdf('champ_data.h5', key='champ_df')
        champ_df.to_csv(os.path.join(opt.output_dir,'champ_data.csv'))
        del champ_df
    #we delete them and re-load later, just as an extra precaution
    print("All done, now merging them.")
    ######## MERGING ########

    columns=['Time (UTC)', 'alt [m]','lon [deg]', 'lat [deg]', 'lst [h]',
        'arglat [deg]', 'dens_x [kg/m3]','satellite_name']

    data_champ=pd.read_csv(os.path.join(opt.output_dir,'champ_data.csv'))
    data_champ['satellite_name']=['champ']*data_champ.shape[0]
    data_champ=data_champ[columns]

    data_goce=pd.read_csv(os.path.join(opt.output_dir,'goce_data.csv'))
    data_goce['satellite_name']=['goce']*data_goce.shape[0]
    data_goce=data_goce[columns]

    data_grace_a=pd.read_csv(os.path.join(opt.output_dir,'grace_a_data.csv'))
    data_grace_a['satellite_name']=['grace_a']*data_grace_a.shape[0]
    data_grace_a=data_grace_a[columns]

    data_grace_b=pd.read_csv(os.path.join(opt.output_dir,'grace_b_data.csv'))
    data_grace_b['satellite_name']=['grace_b']*data_grace_b.shape[0]
    data_grace_b=data_grace_b[columns]

    data_grace_c=pd.read_csv(os.path.join(opt.output_dir,'grace_c_data.csv'))
    data_grace_c['satellite_name']=['grace_c']*data_grace_c.shape[0]
    data_grace_c=data_grace_c[columns]

    data_swarm_a=pd.read_csv(os.path.join(opt.output_dir,'swarm_a_data.csv'))
    data_swarm_a['satellite_name']=['swarm_a']*data_swarm_a.shape[0]
    data_swarm_a=data_swarm_a[columns]

    data_swarm_b=pd.read_csv(os.path.join(opt.output_dir,'swarm_b_data.csv'))
    data_swarm_b['satellite_name']=['swarm_b']*data_swarm_b.shape[0]
    data_swarm_b=data_swarm_b[columns]

    data_swarm_c=pd.read_csv(os.path.join(opt.output_dir,'swarm_c_data.csv'))
    data_swarm_c['satellite_name']=['swarm_c']*data_swarm_c.shape[0]
    data_swarm_c=data_swarm_c[columns]

    dfs_satellites=[data_champ,data_goce,data_grace_a,data_grace_b,data_grace_c,data_swarm_a,data_swarm_b,data_swarm_c]
    merged_df = pd.concat(dfs_satellites, ignore_index=True)
    print('merging done.')
    del dfs_satellites
    print(f'columns are: {merged_df.columns}')
    merged_df.sort_values(by='Time (UTC)', inplace=True)
    print('sorting by dates done.')
    merged_df.reset_index(drop=True,inplace=True)
    merged_df.reset_index(drop=True,inplace=True)

    merged_df.replace([np.inf, -np.inf], np.nan,inplace=True)
    merged_df.dropna(axis=0,inplace=True)
    merged_df.reset_index(drop=True,inplace=True)
    print(merged_df.columns)

    #we first drop the unused columns:
    merged_df.drop(['arglat [deg]', 'lst [h]'], axis=1, inplace=True)
    print(merged_df.columns)

    val=pd.DatetimeIndex(merged_df['Time (UTC)'])
    merged_df['year']=val.year
    merged_df['doy']=val.day_of_year
    merged_df['sid']=val.hour*3600+val.minute*60+val.second+val.microsecond/1e6
    merged_df=merged_df[merged_df['dens_x [kg/m3]']>1e-14]#I exclude very low values of the density (can be errors in the POD)

    #changing the column names according to the established convention:
    dict_columns={
                'Time (UTC)': 'all__dates_datetime__',
                'doy':'all__day_of_year__[d]',
                'year':'all__year__[y]',
                'sid':'all__seconds_in_day__[s]',
                'dens_x [kg/m3]': 'tudelft_thermo__ground_truth_thermospheric_density__[kg/m**3]',
                'lon [deg]': 'tudelft_thermo__longitude__[deg]',
                'lat [deg]': 'tudelft_thermo__latitude__[deg]',
                'alt [m]':'tudelft_thermo__altitude__[m]',
                'satellite_name':'tudelft_thermo__satellite__',
                }
    merged_df.rename(columns=dict_columns, inplace=True)
    merged_df.sort_values(by='all__dates_datetime__', inplace=True)
    merged_df.to_csv(os.path.join(opt.output_dir,'satellites_data.csv'),index=False)


if __name__ == "__main__":
    time_start = time.time()
    process_tudelft_data()
    print('\nTotal duration: {}'.format(time.time() - time_start))
    sys.exit(0)
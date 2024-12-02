import numpy as np
import os
import argparse
import datetime
import pandas as pd
import time
import sys
import netCDF4 as nc
import cftime

from io import StringIO
from pyfiglet import Figlet
from termcolor import colored


def get_str_irr(mission,wavelength):
    if mission < 16:
        if wavelength==30.4:
            str_irr = 'irr_chanB'
        elif wavelength==121.6:
            str_irr = 'irr_chanE_uncorr'
        str_irr_flag = 'irr_'+str(int(wavelength*10))+'_flag'
    else:
        str_irr = 'irr_'+str(int(wavelength*10))
        str_irr_flag = 'irr_'+str(int(wavelength*10))+'_flag'

    return str_irr,str_irr_flag

def get_goes_data(mission, year, str_irr, str_irr_flag, input_dir):
        goes_ds = nc.Dataset(os.path.join(input_dir,'goes'+str(mission)+'_y'+str(year)+'.nc'))
        time_ds = goes_ds.variables['time']
        t = cftime.num2pydate(time_ds[:],time_ds.units)
        irr = goes_ds[str_irr][:]
        firr = goes_ds[str_irr_flag][:]

        return t, irr, firr

def process_goes_data():
    print('GOES Data Processing')
    f = Figlet(font='5lineoblique')
    print(colored(f.renderText('KARMAN 2.0'), 'red'))
    f = Figlet(font='digital')
    print(colored(f.renderText("GOES Data Processing"), 'blue'))
    #print(colored(f'Version {karman.__version__}\n','blue'))

    parser = argparse.ArgumentParser(description='GOES Data Processing', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--input_dir', type=str, default='../../data/goes_data_raw', help='Input directory of raw data')
    parser.add_argument('--output_dir', type=str, default='../../data/goes_data', help='Output directory of processed data')

    opt = parser.parse_args()

    #create output directory if it does not exist
    os.makedirs(opt.output_dir, exist_ok=True)

    all_wavelengths = [25.6, 28.4, 30.4, 117.5, 121.6, 133.5, 140.5]    #nm
    num_wave = np.size(all_wavelengths)

    for iwave in range(num_wave):
        wavelength = all_wavelengths[iwave]
        print(f'Processing GOES irradiance data at {wavelength}nm')

        strIrradiance = 'goes__irradiance_'+str(int(wavelength*10))+'nm___[W/m2]'
        strFlag = 'source__gaps_flag__'#'goes__irradiance_'+str(int(wavelength*10))+'nm___flag'

        outputstr = 'goes_irradiance_'+str(int(wavelength*10))+'nm_sw.csv'

        dict_set={'all__dates_datetime__':[],
                strIrradiance:[],
                strFlag:[],}


        if wavelength==30.4 or wavelength==121.6:
            all_years = range(2010,2024+1)
            missions_years = [[15], [15], [15], [15], [15], [15], [15], [16, 15], [16, 17, 15], [16, 17, 15], [16, 17, 15], [16, 17], [16, 17, 18], [16, 17, 18], [16, 18]]
        else:
            all_years =range(2017,2024+1)
            missions_years = [[16], [16, 17], [16, 17], [16, 17], [16, 17], [16, 17, 18], [16, 17, 18], [16, 18]]
            

        prev_irradiance = -1
        prev_flag = 24*60

        for year in all_years:
            print(f'-----> Year {year}')
            all__dates = pd.date_range(start=datetime.datetime(year,1,1,0,0),end=datetime.datetime(year,12,31,23,59),freq="1min")
            num_dates = np.size(all__dates)
            
            missions = missions_years[year-all_years[0]]
            num_missions = np.size(missions)

            goes__irradiance = -1*np.ones([num_dates,num_missions+1])
            goes__irradiance__flag = -1*np.ones([num_dates,num_missions+1])

            for imission in range(num_missions):
                str_irr,str_irr_flag = get_str_irr(missions[imission],wavelength)
                time_stamp, irr, firr = get_goes_data(missions[imission], year, str_irr, str_irr_flag,opt.input_dir)


                istart = np.where(all__dates==time_stamp[0])[0][0]
                iend = np.where(all__dates==time_stamp[-1])[0][0]

                goes__irradiance[istart:iend+1,imission] = irr
                goes__irradiance__flag[istart:iend+1,imission] = firr

            for itime in range(num_dates):
                imission = 0
                irr_check = -1

                while imission<num_missions and irr_check<0:
                    irr = goes__irradiance[itime,imission]

                    if goes__irradiance__flag[itime,imission]==0:
                        if not(missions[imission]<16 and num_missions>1):
                            goes__irradiance__flag[itime,-1] = 0
                            goes__irradiance[itime,-1] = irr

                            prev_irradiance = irr
                            prev_flag = 0

                            irr_check = 1

                    imission +=1

                if irr_check<0:
                    goes__irradiance[itime,-1] = prev_irradiance
                    prev_flag += 1

                    if prev_flag < 30:
                        goes__irradiance__flag[itime] = 1
                    elif prev_flag < 120:
                        goes__irradiance__flag[itime] = 2
                    elif prev_flag < 24*60:
                        goes__irradiance__flag[itime] = 3
                    else:
                        goes__irradiance__flag[itime] = 4

            #Storage
            dict_set['all__dates_datetime__'].extend(all__dates)
            dict_set[strIrradiance].extend(goes__irradiance[:,-1])
            dict_set[strFlag].extend(goes__irradiance__flag[:,-1])


        first_index = list(x > 0 for x in dict_set[strIrradiance]).index(True)
        last_index = list(x > datetime.datetime.now() for x in dict_set['all__dates_datetime__']).index(True)

        dict_set['all__dates_datetime__'] = dict_set['all__dates_datetime__'][first_index:last_index]
        dict_set[strIrradiance] = dict_set[strIrradiance][first_index:last_index]
        dict_set[strFlag] = dict_set[strFlag][first_index:last_index]

        df_goes=pd.DataFrame(dict_set)
        df_goes.sort_values(by=['all__dates_datetime__'],ascending=True,inplace=True)
        print('Save to csv')
        df_goes.to_csv(os.path.join(opt.output_dir,outputstr),index=False)

if __name__ == "__main__":
    time_start = time.time()
    process_goes_data()
    print('\nTotal duration: {}'.format(time.time() - time_start))
    sys.exit(0)
import numpy as np
import os
import argparse
import datetime
import pandas as pd
import time
import sys


from io import StringIO
from tqdm import tqdm
from pyfiglet import Figlet
from termcolor import colored

def create_flag(data,column_nan):
    """
    This function creates an extra column in the data that is a flag.
    This flag is:
    - 0 if the data is not nan
    - 1 if the data is nan and the previous data is less than 30 minutes
    - 2 if the data is nan and the previous data is between 30 minutes and 2 hours
    - 3 if the data is nan and the previous data is between 2 hours and 24 hours
    - 4 if the data is nan and the previous data is more than 24 hours

    This is handy to have a flag that can be used to filter the data, after we forward fill.

    Pars:
    
        - data (`pd.DataFrame`): pandas dataframe
        - column_nan (`str`): column name that we want to create the flag for
    
    Returns:

        - `np.array` of integers
    """
    times_ffilled=data['all__dates_datetime__'].where(pd.notna(data[column_nan])).ffill()
    #delta times (in minutes)
    delta_times=(data['all__dates_datetime__']-times_ffilled).dt.total_seconds()/60

    mask_1=(delta_times<30) & (delta_times>0)
    mask_2=(delta_times>=30) & (delta_times<120)
    mask_3=(delta_times>=120) & (delta_times<60*24)
    mask_4=(delta_times>=60*24)

    boolean_delta_times=np.zeros((len(delta_times),),dtype=int)
    boolean_delta_times[mask_1]=1
    boolean_delta_times[mask_2]=2
    boolean_delta_times[mask_3]=3
    boolean_delta_times[mask_4]=4
    return boolean_delta_times

def process_omni_data():
    print('Omniweb Data Processing')
    f = Figlet(font='5lineoblique')
    print(colored(f.renderText('KARMAN 2.0'), 'red'))
    f = Figlet(font='digital')
    print(colored(f.renderText("Omniweb Data Processing"), 'blue'))
    #print(colored(f'Version {karman.__version__}\n','blue'))

    parser = argparse.ArgumentParser(description='Omniweb Data Processing', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--input_dir', type=str, default='../../data/omniweb_data_raw', help='Input directory of raw data')
    parser.add_argument('--output_dir', type=str, default='../../data/omniweb_data', help='Output directory of processed data')

    opt = parser.parse_args()

    #create output directory if it does not exist
    os.makedirs(opt.output_dir, exist_ok=True)

    #useful to identify what are the NaN in omniweb data
    dic_invalid={}
    dic_invalid['1 RMS Min_var']=99.99
    dic_invalid['2 Time btwn observations,sec']=999999.00
    dic_invalid['3 Field magnitude average, nT']=9999.99
    dic_invalid['4 BX, nT (GSE, GSM)']=9999.99
    dic_invalid['5 BY, nT (GSE)']=9999.99
    dic_invalid['6 BZ, nT (GSE)']=9999.99
    dic_invalid['7 BY, nT (GSM)']=9999.99
    dic_invalid['8 BZ, nT (GSM)']=9999.99
    dic_invalid['9 RMS SD B scalar, nT']=9999.99
    dic_invalid['10 RMS SD field vector, nT']=9999.99
    dic_invalid['11 Speed, km/s']=99999.90
    dic_invalid['12 Vx Velocity,km/s']=99999.90
    dic_invalid['13 Vy Velocity, km/s']=99999.90
    dic_invalid['14 Vz Velocity, km/s']=99999.90
    dic_invalid['15 Proton Density, n/cc']=999.99
    dic_invalid['16 Proton Temperature, K']=9999999.00
    dic_invalid['17 Flow pressure, nPa']=99.99
    dic_invalid['18 Electric field, mV/m']=999.99
    dic_invalid['19 Plasma beta']=999.99
    dic_invalid['20 Alfven mach number']=999.90
    dic_invalid['21 S/C, Xgse,Re']=9999.99
    dic_invalid['22 S/C, Ygse,Re']=9999.99
    dic_invalid['23 S/c, Zgse,Re']=9999.99
    dic_invalid['24 BSN location, Xgse,Re']=9999.99
    dic_invalid['25 BSN location, Ygse,Re']=9999.99
    dic_invalid['26 BSN location, Zgse,Re']=9999.99
    dic_invalid['27 AE-index, nT']=99999
    dic_invalid['28 AL-index, nT']=99999
    dic_invalid['29 AU-index, nT']=99999
    #and this is another useful dictionary for the renaming part:
    dict_names={
        'time': 'all__dates_datetime__', 
        '1 RMS Min_var': 'omniweb__rms_min_var__',
        '2 Time btwn observations,sec':'omniweb__rms_min_var__[s]', 
        '3 Field magnitude average, nT':'omniweb__field_magnitude_average__nT',
        '4 BX, nT (GSE, GSM)': 'omniweb__bx_gse__[nT]', 
        '5 BY, nT (GSE)': 'omniweb__by_gse__[nT]', 
        '6 BZ, nT (GSE)': 'omniweb__bz_gse__[nT]',
        '7 BY, nT (GSM)': 'omniweb__by_gsm__[nT]', 
        '8 BZ, nT (GSM)': 'omniweb__bz_gsm__[nT]', 
        '9 RMS SD B scalar, nT':'omniweb__rms_sd_b_scalar__[nT]',
        '10 RMS SD field vector, nT':'omniweb__rms_sd_field_vector__[nT]', 
        '11 Speed, km/s':'omniweb__speed__[km/s]', 
        '12 Vx Velocity,km/s':'omniweb__vx_velocity__[km/s]',
        '13 Vy Velocity, km/s':'omniweb__vy_velocity__[km/s]', 
        '14 Vz Velocity, km/s':'omniweb__vz_velocity__[km/s]',
        '15 Proton Density, n/cc':'omniweb__proton_density__[n/cc]',
        '16 Proton Temperature, K':'omniweb__proton_temperature__[K]',
        '17 Flow pressure, nPa':'omniweb__flow_pressure__[nPa]', 
        '18 Electric field, mV/m':'omniweb__electric_field__[mV/m]', 
        '19 Plasma beta':'omniweb__plasma_beta__',
        '20 Alfven mach number':'omniweb__alfven_mach_number__', 
        '21 S/C, Xgse,Re':'omniweb__sc_xgse__[Re]', 
        '22 S/C, Ygse,Re':'omniweb__sc_ygse__[Re]',
        '23 S/c, Zgse,Re':'omniweb__sc_zgse__[Re]', 
        '24 BSN location, Xgse,Re':'omniweb__bsn_location_xgse__[Re]',
        '25 BSN location, Ygse,Re':'omniweb__bsn_location_ygse__[Re]', 
        '26 BSN location, Zgse,Re':'omniweb__bsn_location_zgse__[Re]',
        '27 AE-index, nT':'omniweb__ae_index__[nT]', 
        '28 AL-index, nT':'omniweb__al_index__[nT]', 
        '29 AU-index, nT':'omniweb__au_index__[nT]', 
        '30 SYM/D, nT':'omniweb__sym_d__[nT]',
        '31 SYM/H, nT':'omniweb__sym_h__[nT]', 
        '32 ASY/D, nT':'omniweb__asy_d__[nT]',
    }

    files=os.listdir(opt.input_dir)
    files=[os.path.join(opt.input_dir,f) for f in files if f.endswith('.txt') and f.startswith('data_')]


    # Suppress the SettingWithCopyWarning
    pd.options.mode.chained_assignment = None

    for file_path in tqdm(files):
            
        # Read the file, skipping the initial non-data rows
        with open(file_path, 'r') as file:
            lines = file.readlines()

        # Find column names:
        start_=None
        for i,line in enumerate(lines):
            if line.strip().startswith('1'):
                start_=i
                break
        column_names_not_time=lines[start_:start_+32]
        column_names_not_time=[c.strip() for c in column_names_not_time]
        # Find the index of the line that starts the column headers
        start_idx = None
        for i, line in enumerate(lines):
            if line.startswith("YYYY DOY HR MN"):
                start_idx = i
                break

        # Find the index of the line that ends the data
        end_idx = None
        for i in range(len(lines)-1, -1, -1):
            if lines[i].strip().startswith("</pre>"):
                end_idx = i
                break

        # If we found the start and end indices, process further
        if start_idx is not None and end_idx is not None:
            # Extract the header line
            header_line = lines[start_idx].strip()
            column_names = header_line.split()[:4]+column_names_not_time

            # Extract the data lines
            data_lines = lines[start_idx + 1:end_idx]

            # Convert the list of lines into a single string
            data_str = ''.join(data_lines)

            # Read the data into a DataFrame
            data = pd.read_csv(StringIO(data_str), sep=r"\s+", names=column_names)
            print('Data prepared in pandas dataframe')

        else:
            print("Header row or end tag not found in the file.")
        #we first substitute the year/doy/hour/min with datetime:
        years=data['YYYY'].values
        doys=data['DOY'].values+data['HR'].values/24+data['MN'].values/(24*60)
        datetime_list=[datetime.datetime(year,1,1)+datetime.timedelta(doy-1) for year,doy in zip(years,doys)]
        datetime_list=pd.to_datetime(datetime_list)
        data['time']=datetime_list
        data.drop(columns=['YYYY','DOY','HR','MN'],inplace=True)
        #now we need to deal with the invalid data (which we transform into NaN)
        for key in dic_invalid.keys():
            data.loc[data[key]==dic_invalid[key],key]=np.nan
        #now we need to adjust first the column names
        data.rename(columns=dict_names,inplace=True)
        #and then we create three sub-datasets. One for the omni magnetic field data, one for the solar wind velocity, one for the indices
        data_omni_magnetic_field=data[['all__dates_datetime__','omniweb__bx_gse__[nT]','omniweb__by_gse__[nT]','omniweb__bz_gse__[nT]']]
        data_omni_solar_wind_velocity=data[['all__dates_datetime__','omniweb__speed__[km/s]','omniweb__vx_velocity__[km/s]','omniweb__vy_velocity__[km/s]','omniweb__vz_velocity__[km/s]']]
        data_omni_indices=data[['all__dates_datetime__','omniweb__ae_index__[nT]','omniweb__al_index__[nT]','omniweb__au_index__[nT]','omniweb__sym_d__[nT]','omniweb__sym_h__[nT]','omniweb__asy_d__[nT]']]
        #finally we need to perform the forward filling and add a column of flags indicating the time difference
        boolean_delta_times=create_flag(data_omni_magnetic_field,'omniweb__bx_gse__[nT]')
        data_omni_magnetic_field.loc[:,'source__gaps_flag__']=boolean_delta_times
        data_omni_magnetic_field.ffill(inplace=True)

        boolean_delta_times=create_flag(data_omni_solar_wind_velocity,'omniweb__speed__[km/s]')
        data_omni_solar_wind_velocity.loc[:,'source__gaps_flag__']=boolean_delta_times
        data_omni_solar_wind_velocity.ffill(inplace=True)

        boolean_delta_times=create_flag(data_omni_indices,'omniweb__al_index__[nT]')
        data_omni_indices.loc[:,'source__gaps_flag__']=boolean_delta_times
        data_omni_indices.ffill(inplace=True)

        #we can now save them:
        file_date=file_path.split('_')[-2]+'_'+file_path.split('_')[-1].split('.')[0]
        print(f'saving to file, file date: {file_date}')
        data_omni_magnetic_field.to_csv(os.path.join(opt.output_dir,f'magnetic_field_omni_{file_date}.csv'),index=False)
        data_omni_solar_wind_velocity.to_csv(os.path.join(opt.output_dir,f'solar_wind_velocity_omni_{file_date}.csv'),index=False)
        data_omni_indices.to_csv(os.path.join(opt.output_dir,f'indices_omni_{file_date}.csv'),index=False)
        print('Done')

if __name__ == "__main__":
    time_start = time.time()
    process_omni_data()
    print('\nTotal duration: {}'.format(time.time() - time_start))
    sys.exit(0)
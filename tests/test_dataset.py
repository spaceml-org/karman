import pyatmos
import unittest
import pandas as pd
import numpy as np
import os
from io import StringIO
import karman
from karman import ThermosphericDensityDataset
#idxs=np.random.choice(range(0,len(data)),20)
#data.loc[idxs].to_csv("sample_data_test.csv")
file_content="""
all__day_of_year__[d],all__year__[y],all__seconds_in_day__[s],tudelft_thermo__altitude__[m],tudelft_thermo__longitude__[deg],tudelft_thermo__latitude__[deg],tudelft_thermo__local_solar_time__[h],space_environment_technologies__f107_average__,space_environment_technologies__f107_obs__,celestrack__ap_average__,celestrack__ap_h_0__,celestrack__ap_h_1__,celestrack__ap_h_2__,celestrack__ap_h_3__,celestrack__ap_h_4__,celestrack__ap_h_5__,celestrack__ap_h_6__,all__sun_right_ascension__[rad],all__sun_declination__[rad],all__sidereal_time__[rad],space_environment_technologies__s107_obs__,space_environment_technologies__s107_average__,space_environment_technologies__m107_obs__,space_environment_technologies__m107_average__,space_environment_technologies__y107_obs__,space_environment_technologies__y107_average__,JB08__d_st_dt__[K],tudelft_thermo__ground_truth_thermospheric_density__[kg/m**3],NRLMSISE00__thermospheric_density__[kg/m**3],JB08__thermospheric_density__[kg/m**3],tudelft_thermo__satellite__,all__dates_datetime__
8395602,197.0,2009.0,13275.0,317362.625,131.29241943359375,20.25897789001465,12.440327644348145,67.9000015258789,66.5,2.0,2.0,3.0,2.0,2.0,4.0,4.375,11.375,2.0155909061431885,0.3730264604091644,2.106971263885498,52.59999847412109,54.59999847412109,63.400001525878906,63.79999923706055,55.79999923706055,58.400001525878906,17.0,3.656764127613066e-12,5.348937438809376e-12,4.264762154537749e-12,champ,2009-07-16 03:41:15
10993118,317.0,2004.0,25637.0,508700.71875,65.61274719238281,-58.70895385742188,11.495572090148926,105.9000015258789,94.9000015258789,30.0,30.0,39.0,32.0,9.0,15.0,40.0,206.0,3.974699497222901,-0.3104558289051056,3.9128050804138175,110.9000015258789,103.6999969482422,102.1999969482422,107.0999984741211,126.8000030517578,113.3000030517578,130.8352813720703,3.3697532069408564e-13,6.556180836940484e-13,7.429702469571864e-13,grace,2004-11-12 07:07:17
31735563,254.0,2020.0,3660.0,455377.53125,-81.2591323852539,-53.71056365966797,-4.400609016418457,71.5999984741211,69.69999694824219,2.0,2.0,0.0,0.0,0.0,2.0,1.5,2.75,2.9409708976745605,0.0861753225326538,4.948400497436523,58.5,59.400001525878906,72.0999984741211,75.19999694824219,66.9000015258789,68.4000015258789,0.0,1.1638571508452988e-13,1.4470035955040097e-13,1.0162580683665406e-13,swarm_a,2020-09-10 01:01:00
11439106,243.0,2005.0,82517.0,469441.90625,-22.77816200256348,88.95315551757812,21.40284538269043,89.19999694824219,86.0,49.0,49.0,5.0,6.0,12.0,2.0,1.5,8.5,2.795428276062012,0.1460521519184112,5.257869243621826,100.6999969482422,94.6999969482422,108.0999984741211,90.4000015258789,107.0999984741211,97.4000015258789,266.1005554199219,1.299797075526854e-12,8.347816795456486e-13,1.1748657602339565e-12,grace,2005-08-31 22:55:17
19092264,328.0,2011.0,18510.0,274289.875,-145.8984375,-47.71026992797852,-4.584895610809326,146.10000610351562,140.0,7.0,7.0,7.0,7.0,5.0,4.0,7.125,7.25,4.176002025604248,-0.3566963672637939,6.17919397354126,152.6999969482422,155.5,148.1999969482422,137.10000610351562,143.39999389648438,142.6999969482422,50.0,4.200642958473999e-11,4.9156522602400437e-11,4.9921643208161726e-11,goce,2011-11-24 05:08:30
5603822,308.0,2006.0,27046.0,357646.71875,-1.053683876991272,47.18717956542969,7.442532062530518,81.80000305175781,87.4000015258789,7.0,7.0,7.0,2.0,5.0,5.0,10.5,6.125,3.825917482376098,-0.2674887180328369,2.706089496612549,77.30000305175781,76.0,75.80000305175781,75.0999984741211,78.5999984741211,78.9000015258789,50.0,1.4494989410140937e-12,3.0187601550435428e-12,2.189122581125469e-12,champ,2006-11-04 07:30:46
28070567,61.0,2017.0,39480.0,459053.71875,66.78919982910156,-42.19841766357422,15.419279098510742,78.0,81.0,32.0,32.0,56.0,67.0,56.0,32.0,18.75,8.5,5.990438461303711,-0.1244643405079841,0.5535926222801208,71.80000305175781,69.5,73.0,67.69999694824219,93.9000015258789,90.4000015258789,115.0,7.529759151261861e-13,7.351678402330032e-13,7.172545435189803e-13,swarm_a,2017-03-02 10:58:00
35614940,365.0,2015.0,80910.0,511549.21875,-30.41947364807129,-14.135205268859863,20.44703483581543,108.5999984741211,101.5,35.0,35.0,18.0,5.0,3.0,3.0,3.875,4.25,4.896472930908203,-0.4028753042221069,0.8156664371490479,100.5999984741211,110.0999984741211,113.8000030517578,117.6999969482422,121.4000015258789,119.0,194.8000030517578,3.325575492069133e-13,3.0093020631817574e-13,4.808124926848624e-13,swarm_b,2015-12-31 22:28:30
1837920,68.0,2003.0,377.0,412226.40625,-18.667314529418945,65.3176498413086,-1.139765381813049,126.1999969482422,148.3000030517578,11.0,11.0,9.0,9.0,12.0,18.0,7.625,22.375,6.090001106262207,-0.0830425471067428,2.603719472885132,150.8000030517578,130.89999389648438,169.8000030517578,129.6999969482422,145.0,136.8000030517578,94.0,3.20320332758417e-12,2.4971958095115765e-12,3.1684856561381425e-12,champ,2003-03-09 00:06:17
19587400,22.0,2012.0,71340.0,267627.65625,165.1726837158203,-18.289772033691406,30.82817840576172,124.0,141.60000610351562,24.0,24.0,3.0,7.0,7.0,4.0,5.375,3.875,5.31060791015625,-0.3439902365207672,3.909423589706421,135.8000030517578,130.89999389648438,127.0,120.9000015258789,138.60000610351562,128.39999389648438,124.0,3.9196229595361836e-11,4.134798753385738e-11,4.633002662068897e-11,goce,2012-01-22 19:49:00
27923187,10.0,2017.0,11430.0,459115.75,68.0726318359375,-43.06740188598633,7.713175773620605,75.9000015258789,71.69999694824219,10.0,10.0,7.0,9.0,5.0,18.0,14.875,16.125,5.086825370788574,-0.383500337600708,3.936401844024658,67.5,70.5999984741211,61.09999847412109,68.30000305175781,89.80000305175781,91.5,50.0,1.559434276588584e-13,2.952086275084809e-13,1.788392567717012e-13,swarm_a,2017-01-10 03:10:30
4671872,345.0,2005.0,15137.0,367045.15625,-37.679969787597656,-49.097537994384766,1.6927244663238523,86.69999694824219,91.4000015258789,21.0,21.0,22.0,15.0,15.0,7.0,13.375,3.0,4.506189823150635,-0.4013221859931946,1.8390368223190308,94.0999984741211,88.69999694824219,84.80000305175781,86.4000015258789,96.5,89.19999694824219,85.0,2.29374180239772e-12,2.246296681648885e-12,2.8555549851788653e-12,champ,2005-12-11 04:12:17
37191965,201.0,2017.0,83760.0,511337.84375,-51.0412483215332,-21.173986434936523,19.863916397094727,76.4000015258789,73.0999984741211,6.0,6.0,3.0,3.0,5.0,4.0,2.75,14.625,2.099051713943481,0.3582547605037689,4.1343913078308105,65.69999694824219,64.5999984741211,72.30000305175781,75.69999694824219,94.8000030517578,84.30000305175781,50.0,9.591245441094298e-15,6.149703820181389e-14,4.7391059718009116e-14,swarm_b,2017-07-20 23:16:00
3038268,127.0,2004.0,9197.0,380624.40625,-104.91547393798828,21.005260467529297,-4.439642429351807,100.4000015258789,88.5,8.0,8.0,6.0,12.0,18.0,27.0,10.375,7.25,0.7572078704833984,0.2894331812858581,2.7523584365844727,99.9000015258789,106.3000030517578,100.0999984741211,105.4000015258789,114.5999984741211,112.5999984741211,94.0,2.349022452247107e-12,2.585186838904852e-12,3.1668873252954644e-12,champ,2004-05-06 02:33:17
7690154,311.0,2008.0,9706.0,329781.0625,-37.84368896484375,56.89836883544922,0.1731986999511718,68.5,67.69999694824219,1.0,1.0,0.0,2.0,2.0,2.0,1.75,1.125,3.8654744625091553,-0.2796030342578888,0.8427597284317017,51.79999923706055,52.400001525878906,64.0,63.79999923706055,57.900001525878906,57.20000076293945,17.0,1.654325114397448e-12,2.853774725211644e-12,1.939618604618332e-12,champ,2008-11-06 02:41:46
20963057,194.0,2012.0,68180.0,261060.40625,-170.8339385986328,46.187408447265625,7.549959182739258,126.3000030517578,161.6999969482422,10.0,10.0,12.0,15.0,7.0,5.0,10.25,22.625,1.96018648147583,0.3814255893230438,0.7735134959220886,136.89999389648438,126.5,138.5,118.5,151.0,128.89999389648438,67.00555419921875,3.407278972855643e-11,4.225763836074314e-11,4.500814304586598e-11,goce,2012-07-12 18:56:20
9463755,207.0,2010.0,13215.0,268950.28125,150.8467559814453,36.895469665527344,13.72728443145752,78.0999984741211,85.19999694824219,5.0,5.0,7.0,6.0,5.0,5.0,5.375,5.75,2.1853599548339844,0.3404153883457184,2.611724376678467,82.69999694824219,80.30000305175781,80.30000305175781,79.0999984741211,95.4000015258789,89.5999984741211,39.025001525878906,2.3763631687434558e-11,2.6628439037112983e-11,2.5145356283284848e-11,champ,2010-07-26 03:40:15
28769899,304.0,2017.0,76110.0,442104.28125,124.8483428955078,31.198766708374023,29.464889526367188,75.19999694824219,75.5999984741211,2.0,2.0,5.0,3.0,0.0,2.0,2.0,3.25,3.770188331604004,-0.2495868802070617,2.135094165802002,66.69999694824219,63.79999923706055,86.80000305175781,76.5999984741211,88.30000305175781,82.80000305175781,50.0,1.511328632771217e-13,2.058227991253575e-13,2.0920672963095635e-13,swarm_a,2017-10-31 21:08:30
32408128,140.0,2021.0,20340.0,443049.4375,-133.13287353515625,-32.42984390258789,-3.225524425506592,78.9000015258789,77.69999694824219,24.0,24.0,12.0,9.0,5.0,3.0,3.75,10.25,0.9936386346817015,0.348455399274826,3.3112759590148926,68.0,66.4000015258789,88.69999694824219,87.30000305175781,80.80000305175781,83.0,85.0,2.751809468227312e-13,2.797277112633751e-13,3.333624337947122e-13,swarm_a,2021-05-20 05:39:00
39162580,237.0,2019.0,56340.0,519899.625,-17.232406616210938,-46.93542098999024,14.50117301940918,67.4000015258789,66.30000305175781,3.0,3.0,4.0,0.0,3.0,4.0,3.625,4.75,2.685789108276367,0.1885537654161453,3.335899591445923,56.29999923706055,56.5,68.80000305175781,68.19999694824219,74.5,72.4000015258789,46.09999847412109,1.0691014723897621e-13,5.5696139838806384e-14,5.928361913789398e-14,swarm_b,2019-08-25 15:39:00
"""
df = pd.read_csv(StringIO(file_content))

class DatasetTestCases(unittest.TestCase):
    def test_nrlmsise00_output(self):       
        sw_nrlmsise00_file=pyatmos.download_sw_nrlmsise00()
        swdata_nrlmsise = pyatmos.read_sw_nrlmsise00(sw_nrlmsise00_file)
        alts=df['tudelft_thermo__altitude__[m]'].values
        lats=df['tudelft_thermo__latitude__[deg]'].values
        lons=df['tudelft_thermo__longitude__[deg]'].values
        dates=df['all__dates_datetime__'].values
        ground_truth_rhos=df['tudelft_thermo__ground_truth_thermospheric_density__[kg/m**3]'].values
        nrlmsise00_rhos=df['NRLMSISE00__thermospheric_density__[kg/m**3]'].values
        for i in range(len(df)):
            rho_nrlmisse00=pyatmos.nrlmsise00(dates[i],(lats[i],lons[i],alts[i]*1e-3),swdata_nrlmsise).rho
            self.assertAlmostEqual(rho_nrlmisse00,nrlmsise00_rhos[i],places=6)
    
    def test_jb08_output(self):
        df = pd.read_csv(StringIO(file_content))
        sw_jb2008_file=pyatmos.download_sw_jb2008()
        swdata_jb08 = pyatmos.read_sw_jb2008(sw_jb2008_file)
        alts=df['tudelft_thermo__altitude__[m]'].values
        lats=df['tudelft_thermo__latitude__[deg]'].values
        lons=df['tudelft_thermo__longitude__[deg]'].values
        dates=df['all__dates_datetime__'].values
        jb2008_rhos=df['JB08__thermospheric_density__[kg/m**3]'].values
        for i in range(len(df)):
            rho_jb08=pyatmos.jb2008(dates[i],(lats[i],lons[i],alts[i]*1e-3),swdata_jb08).rho
            self.assertAlmostEqual(rho_jb08,jb2008_rhos[i],places=6)
    def test_dataset(self):
        #I construct a dataset using the Karman dataset class (small version of the dataset, to avoid long tests):
        omni_resolution=1
        fism2_flare_stan_bands_resolution=1
        fism2_daily_stan_bands_resolution=60*24
        dataset=ThermosphericDensityDataset(
            directory=os.path.join(os.getcwd(),"tests/data"),
            include_omni=True,
            include_daily_stan_bands=True,
            include_flare_stan_bands=True,
            thermo_scaler=None,
            min_date=pd.to_datetime('2004-02-01 00:00:00'),
            max_date=pd.to_datetime('2004-02-03 00:00:00'),
            lag_minutes_omni=0,
            lag_minutes_fism2_flare_stan_bands = 0,
            lag_minutes_fism2_daily_stan_bands = 0,
            create_cyclical_features=False,
            omni_resolution=omni_resolution,
            fism2_flare_stan_bands_resolution=fism2_flare_stan_bands_resolution,
            fism2_daily_stan_bands_resolution=fism2_daily_stan_bands_resolution

        )
        #I load the original data:
        ############# INSTANTANEOUS FEATURES DATA ################
        data_thermo=pd.read_hdf(os.path.join(os.getcwd(),"tests/data/jb08_nrlmsise_all_v1/jb08_nrlmsise_all.h5"))
        ############# TIME SERIES DATA ################
        #Omni data:
        data_omni=pd.read_hdf(os.path.join(os.getcwd(),"tests/data/data_omniweb_v1/omniweb_1min_data_2001_2022.h5"))
        data_omni.index=data_omni['all__dates_datetime__']
        data_omni=data_omni.resample(f'{omni_resolution}T').ffill()
        for column in data_omni.columns:
            quantile = data_omni[column].quantile(0.998)
            more_than = data_omni[column] >= quantile
            data_omni.loc[more_than,column] = None
        # We replace NaNs and +/-inf by interpolating them away.
        data_omni = data_omni.replace([np.inf, -np.inf], None)
        data_omni = data_omni.interpolate(method='pad')
        data_fism2_flare=pd.read_hdf(os.path.join(os.getcwd(),"tests/data/fism2_flare_stan_bands.h5"))
        data_fism2_flare.index=data_fism2_flare['all__dates_datetime__']
        data_fism2_flare=data_fism2_flare.resample(f'{fism2_flare_stan_bands_resolution}T').ffill()
        data_fism2_flare.drop_duplicates(inplace=True)
        for column in data_fism2_flare.columns:
            quantile = data_fism2_flare[column].quantile(0.998)
            more_than = data_fism2_flare[column] >= quantile
            data_fism2_flare.loc[more_than,column] = None
        # We replace NaNs and +/-inf by interpolating them away.
        data_fism2_flare = data_fism2_flare.replace([np.inf, -np.inf], None)
        data_fism2_flare = data_fism2_flare.interpolate(method='pad')
        #now fism2 daily:
        data_fism2_daily=pd.read_hdf(os.path.join(os.getcwd(),"tests/data/fism2_daily_stan_bands.h5"))
        data_fism2_daily.index=data_fism2_daily['all__dates_datetime__']
        data_fism2_daily=data_fism2_daily.resample(f'{fism2_daily_stan_bands_resolution}T').ffill()
        data_fism2_daily.drop_duplicates(inplace=True)
        for column in data_fism2_daily.columns:
            quantile = data_fism2_daily[column].quantile(0.998)
            more_than = data_fism2_daily[column] >= quantile
            data_fism2_daily.loc[more_than,column] = None
        # We replace NaNs and +/-inf by interpolating them away.
        data_fism2_daily = data_fism2_daily.replace([np.inf, -np.inf], None)
        data_fism2_daily = data_fism2_daily.interpolate(method='pad')

        cols_thermo=data_thermo.columns
        data_thermo = data_thermo.drop(columns=dataset.features_to_exclude_thermo).values
        cols_fism2_flare=data_fism2_flare.columns
        data_fism2_flare = data_fism2_flare.drop(columns=dataset.features_to_exclude_fism2_flare_stan_bands).values
        cols_fism2_daily=data_fism2_daily.columns
        data_fism2_daily = data_fism2_daily.drop(columns=dataset.features_to_exclude_fism2_daily_stan_bands).values
        cols_omni=data_omni.columns
        data_omni= data_omni.drop(columns=dataset.features_to_exclude_omni).values

        data_thermo_reconstructed=[]
        data_fism2_flare_reconstructed=[]
        data_fism2_daily_reconstructed=[]
        data_omni_reconstructed=[]
        data_thermo_reconstructed=[]
        for i in range(len(dataset)):
            data_thermo_reconstructed.append(dataset[i]['instantaneous_features'].numpy())
            data_fism2_flare_reconstructed.append(dataset[i]['fism2_flare_stan_bands'].numpy())
            data_fism2_daily_reconstructed.append(dataset[i]['fism2_daily_stan_bands'].numpy())
            data_omni_reconstructed.append(dataset[i]['omni'].numpy())
        data_thermo_reconstructed=np.stack(data_thermo_reconstructed)
        data_omni_reconstructed=np.stack(data_omni_reconstructed).squeeze()
        data_fism2_daily_reconstructed=np.stack(data_fism2_daily_reconstructed).squeeze()
        data_fism2_flare_reconstructed=np.stack(data_fism2_flare_reconstructed).squeeze()
        
        # I unnormalize the dataset and compare it with the original one:
        data_thermo_reconstructed_unnormalized=dataset.data_thermo['scaler'].inverse_transform(data_thermo_reconstructed)
        data_fism2_flare_reconstructed_unnormalized=dataset.time_series_data['fism2_flare_stan_bands']['scaler'].inverse_transform(data_fism2_flare_reconstructed)
        data_fism2_daily_reconstructed_unnormalized=dataset.time_series_data['fism2_daily_stan_bands']['scaler'].inverse_transform(data_fism2_daily_reconstructed)
        data_omni_reconstructed_unnormalized=dataset.time_series_data['omni']['scaler'].inverse_transform(data_omni_reconstructed)
        #I extract the unique elements:
        vv=[]
        for i in range(len(data_omni_reconstructed_unnormalized)-1):
            if i==0:
                vv.append(data_omni_reconstructed_unnormalized[i,:])
            elif np.allclose(data_omni_reconstructed_unnormalized[i,:],data_omni_reconstructed_unnormalized[i+1,:])!=True:
                vv.append(data_omni_reconstructed_unnormalized[i+1,:])
        data_omni_reconstructed_unnormalized=np.stack(vv)
        vv_2=[]
        for i in range(len(data_fism2_flare_reconstructed_unnormalized)-1):
            if i==0:
                vv_2.append(data_fism2_flare_reconstructed_unnormalized[i,:])
            elif np.allclose(data_fism2_flare_reconstructed_unnormalized[i,:],data_fism2_flare_reconstructed_unnormalized[i+1,:])!=True:
                vv_2.append(data_fism2_flare_reconstructed_unnormalized[i+1,:])
        data_fism2_flare_reconstructed_unnormalized=np.stack(vv_2)
        vv_3=[]
        for i in range(len(data_fism2_daily_reconstructed_unnormalized)-1):
            if i==0:
                vv_3.append(data_fism2_daily_reconstructed_unnormalized[i,:])
            elif np.allclose(data_fism2_daily_reconstructed_unnormalized[i,:],data_fism2_daily_reconstructed_unnormalized[i+1,:])!=True:
                vv_3.append(data_fism2_daily_reconstructed_unnormalized[i+1,:])
        data_fism2_daily_reconstructed_unnormalized=np.stack(vv_3)
        
        self.assertTrue(np.allclose(data_thermo,data_thermo_reconstructed_unnormalized,rtol=1e-13,atol=1e-13))
        self.assertTrue(np.allclose(data_omni[:-1,:],data_omni_reconstructed_unnormalized,rtol=1e-13,atol=1e-13))
        self.assertTrue(np.allclose(data_fism2_flare[:-1,:],data_fism2_flare_reconstructed_unnormalized,rtol=1e-13,atol=1e-13))
        self.assertTrue(np.allclose(data_fism2_daily[:-1,:],data_fism2_daily_reconstructed_unnormalized,rtol=1e-13,atol=1e-13))
    
    def test_cycling_features(self):
        from sklearn.preprocessing import MinMaxScaler
        omni_resolution=1
        fism2_flare_stan_bands_resolution=1
        fism2_daily_stan_bands_resolution=60*24
        dataset=ThermosphericDensityDataset(
            directory=os.path.join(os.getcwd(),"tests/data"),
            include_omni=True,
            include_daily_stan_bands=True,
            include_flare_stan_bands=True,
            thermo_scaler=None,
            min_date=pd.to_datetime('2004-02-01 00:00:00'),
            max_date=pd.to_datetime('2004-02-03 00:00:00'),
            lag_minutes_omni=0,
            lag_minutes_fism2_flare_stan_bands = 0,
            lag_minutes_fism2_daily_stan_bands = 0,
            create_cyclical_features=True,
            omni_resolution=omni_resolution,
            fism2_flare_stan_bands_resolution=fism2_flare_stan_bands_resolution,
            fism2_daily_stan_bands_resolution=fism2_daily_stan_bands_resolution

        )
        cyclical_features=['all__day_of_year__[d]',
                            'all__seconds_in_day__[s]',
                            'all__sun_right_ascension__[rad]',
                            'all__sun_declination__[rad]',
                            'all__sidereal_time__[rad]',
                            'tudelft_thermo__longitude__[deg]',
                            'tudelft_thermo__local_solar_time__[h]']
        data_unnormalized=dataset.data_thermo['data']
        data={}
        for feature in cyclical_features:
            vals=data_unnormalized[feature].values
            if feature.find('[rad]')!=-1:
                sin_vals=np.sin(vals)
                cos_vals=np.cos(vals)
            else:        
                vals=2*np.pi*(vals-vals.min())/(vals.max()-vals.min())
                sin_vals=np.sin(vals)
                cos_vals=np.cos(vals)        
            data[feature+'_sin']=sin_vals
            data[feature+'_cos']=cos_vals
            
        data=pd.DataFrame(data)
        for col in data.columns:
            self.assertTrue(np.alltrue(dataset.data_thermo['data'][col].values==data[col].values))
        data_normalized=MinMaxScaler().fit_transform(data)
        self.assertTrue(np.allclose(data_normalized, dataset.data_thermo['data_matrix'][:,-14:]))

        self.assertTrue(1==1)

    def train_validation_test_set_test(self):
        #test that the sum of the three corresponds to the original one,
        #and that there are not repeated elements
        omni_resolution=1
        fism2_flare_stan_bands_resolution=1
        fism2_daily_stan_bands_resolution=60*24
        dataset_test=karman.ThermosphericDensityDataset(
            directory=os.path.join(os.getcwd(),"tests/data"),
            include_omni=True,
            include_daily_stan_bands=True,
            include_flare_stan_bands=True,
            thermo_scaler=None,
            min_date=pd.to_datetime('2004-02-01 00:00:00'),
            max_date=pd.to_datetime('2004-02-03 00:00:00'),
            lag_minutes_omni=0,
            lag_minutes_fism2_flare_stan_bands = 0,
            lag_minutes_fism2_daily_stan_bands = 0,
            create_cyclical_features=True,
            omni_resolution=omni_resolution,
            fism2_flare_stan_bands_resolution=fism2_flare_stan_bands_resolution,
            fism2_daily_stan_bands_resolution=fism2_daily_stan_bands_resolution

        )
        dataset_test._set_indices(test_month_idx=[1], validation_month_idx=[0])
        self.assertTrue(len(dataset_test.test_dataset()==len(dataset_test)))
        
        for i in range(len(dataset_test)):
            self.assertTrue(np.allclose(dataset_test[i]['instantaneous_features'].numpy(), dataset_test.test_dataset()[i]['instantaneous_features'].numpy()))
            self.assertTrue(np.allclose(dataset_test[i]['omni'].numpy(), dataset_test.test_dataset()[i]['omni'].numpy()))
            self.assertTrue(np.allclose(dataset_test[i]['fism2_flare_stan_bands_features'].numpy(), dataset_test.test_dataset()[i]['fism2_flare_stan_bands_features'].numpy()))
            self.assertTrue(np.allclose(dataset_test[i]['fism2_daily_stan_bands_features'].numpy(), dataset_test.test_dataset()[i]['fism2_daily_stan_bands_features'].numpy()))
            self.assertTrue(np.allclose(dataset_test[i]['target'].numpy(), dataset_test.test_dataset()[i]['target'].numpy()))

        dataset_valid=karman.ThermosphericDensityDataset(
            directory=os.path.join(os.getcwd(),"tests/data"),
            include_omni=True,
            include_daily_stan_bands=True,
            include_flare_stan_bands=True,
            thermo_scaler=None,
            min_date=pd.to_datetime('2004-02-01 00:00:00'),
            max_date=pd.to_datetime('2004-02-03 00:00:00'),
            lag_minutes_omni=0,
            lag_minutes_fism2_flare_stan_bands = 0,
            lag_minutes_fism2_daily_stan_bands = 0,
            create_cyclical_features=True,
            omni_resolution=omni_resolution,
            fism2_flare_stan_bands_resolution=fism2_flare_stan_bands_resolution,
            fism2_daily_stan_bands_resolution=fism2_daily_stan_bands_resolution

        )
        dataset_valid._set_indices(test_month_idx=[0], validation_month_idx=[1])
        self.assertTrue(len(dataset_valid.validation_dataset()==len(dataset_valid)))
        
        for i in range(len(dataset_valid)):
            self.assertTrue(np.allclose(dataset_valid[i]['instantaneous_features'].numpy(), dataset_valid.validation_dataset()[i]['instantaneous_features'].numpy()))
            self.assertTrue(np.allclose(dataset_valid[i]['omni'].numpy(), dataset_valid.validation_dataset()[i]['omni'].numpy()))
            self.assertTrue(np.allclose(dataset_valid[i]['fism2_flare_stan_bands_features'].numpy(), dataset_valid.validation_dataset()[i]['fism2_flare_stan_bands_features'].numpy()))
            self.assertTrue(np.allclose(dataset_valid[i]['fism2_daily_stan_bands_features'].numpy(), dataset_valid.validation_dataset()[i]['fism2_daily_stan_bands_features'].numpy()))
            self.assertTrue(np.allclose(dataset_valid[i]['target'].numpy(), dataset_valid.validation_dataset()[i]['target'].numpy()))

        dataset_train=karman.ThermosphericDensityDataset(
            directory=os.path.join(os.getcwd(),"tests/data"),
            include_omni=True,
            include_daily_stan_bands=True,
            include_flare_stan_bands=True,
            thermo_scaler=None,
            min_date=pd.to_datetime('2004-02-01 00:00:00'),
            max_date=pd.to_datetime('2004-02-03 00:00:00'),
            lag_minutes_omni=0,
            lag_minutes_fism2_flare_stan_bands = 0,
            lag_minutes_fism2_daily_stan_bands = 0,
            create_cyclical_features=True,
            omni_resolution=omni_resolution,
            fism2_flare_stan_bands_resolution=fism2_flare_stan_bands_resolution,
            fism2_daily_stan_bands_resolution=fism2_daily_stan_bands_resolution

        )
        dataset_train._set_indices(test_month_idx=[0], validation_month_idx=[1])
        self.assertTrue(len(dataset_train.train_dataset()==len(dataset_train)))
        
        for i in range(len(dataset_train)):
            self.assertTrue(np.allclose(dataset_train[i]['instantaneous_features'].numpy(), dataset_train.train_dataset()[i]['instantaneous_features'].numpy()))
            self.assertTrue(np.allclose(dataset_train[i]['omni'].numpy(), dataset_train.train_dataset()[i]['omni'].numpy()))
            self.assertTrue(np.allclose(dataset_train[i]['fism2_flare_stan_bands_features'].numpy(), dataset_train.train_dataset()[i]['fism2_flare_stan_bands_features'].numpy()))
            self.assertTrue(np.allclose(dataset_train[i]['fism2_daily_stan_bands_features'].numpy(), dataset_train.train_dataset()[i]['fism2_daily_stan_bands_features'].numpy()))
            self.assertTrue(np.allclose(dataset_train[i]['target'].numpy(), dataset_train.train_dataset()[i]['target'].numpy()))


    def test_normalization(self):
        #test the minmax normalize
        self.assertTrue(1==1)
    def test_thermo_scaling(self):
        #test thermospheric density scaling
        self.assertTrue(1==1)
    def test_date_to_index(self):
        self.assertTrue(1==1)
    def test_index_to_date(self):
        self.assertTrue(1==1)

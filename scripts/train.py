import argparse
import os
import pprint
import sys
import time

import datetime
import karman
from karman import FeedForwardDensityPredictor, FullFeatureDensityPredictor, Fism2DailyDensityPredictor, Fism2FlareDensityPredictor, OmniDensityPredictor
import numpy as np
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset
from tqdm import tqdm
import wandb

def run():
    print('Karman training script')
    print(f'Version {karman.__version__}\n')

    parser = argparse.ArgumentParser(description='Karman', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', help='Random number seed', default=1, type=int)
    parser.add_argument('--model', help='Models to use', default='FeedForwardDensityPredictor', choices=['FeedForwardDensityPredictor',
                                                                                                         'FullFeatureDensityPredictor',
                                                                                                         'Fism2DailyDensityPredictor',
                                                                                                         'Fism2FlareDensityPredictor',
                                                                                                         'OmniDensityPredictor'])
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--output_directory', help='Output directory', default='output_directory')
    parser.add_argument('--epochs', '-n', help='Number of epochs', default=10, type=int)
    parser.add_argument('--valid_every', default=1, type=int)
    parser.add_argument('--data_directory', default='/home/jupyter/', type=str)
    parser.add_argument('--learning_rate', help='learning rate to use', default=1e-4, type=float)
    parser.add_argument('--weight_decay', help='Weight decay: optimizer parameter', default=0., type=float)
    parser.add_argument('--optimizer', help='Optimizer to use', default='adam', choices=['adam', 'sgd'])
    parser.add_argument('--load_indices',help='If False, then train, validation and test set are computed on the fly; otherwise, they are loaded', default=True, type=bool)
    parser.add_argument('--lag_minutes_omni', help='Time lag (in minutes) to consider for the OMNIWeb data', default=0, type=float)
    parser.add_argument('--lag_days_fism2_daily', help='Time lag (in days) to consider for the FISM2 daily data', default=0, type=float)
    parser.add_argument('--lag_minutes_fism2_flare', help='Time lag (in minutes) to consider for the FISM2 flare data', default=0, type=float)
    parser.add_argument('--exclude_fism2_daily', action='store_true')
    parser.add_argument('--exclude_fism2_flare', action='store_true')
    parser.add_argument('--exclude_omni', action='store_true')
    parser.add_argument('--test_mode', action='store_true')
    parser.add_argument('--wavelength_bands_to_skip', help='FISM2 Irradiance data wavelengths downsampling proportion: base is 0.1 nm (e.g. 10 means every 1nm)', default=10, type=int)
    parser.add_argument('--omniweb_downsampling_ratio', help='OMNIWeb downsampling proportion in time: base is 1 min (e.g. 10 means every 10 min)', default=10, type=int)
    parser.add_argument('--features_to_exclude_omni', help='Features to exclude for OMNI data', default=['all__dates_datetime__',
                                                                                                        'omniweb__id_for_imf_spacecraft__',
                                                                                                        'omniweb__id_for_sw_plasma_spacecraft__',
                                                                                                        'omniweb__#_of_points_in_imf_averages__',
                                                                                                        'omniweb__#_of_points_in_plasma_averages__',
                                                                                                        'omniweb__percent_of_interpolation__',
                                                                                                        'omniweb__timeshift__[s]',
                                                                                                        'omniweb__rms_timeshift__[s]',
                                                                                                        'omniweb__rms_min_variance__[s**2]',
                                                                                                        'omniweb__time_btwn_observations__[s]',
                                                                                                        'omniweb__rms_sd_b_scalar__[nT]',
                                                                                                        'omniweb__rms_sd_b_field_vector__[nT]',
                                                                                                        'omniweb__flow_pressure__[nPa]',
                                                                                                        'omniweb__electric_field__[mV/m]',
                                                                                                        'omniweb__plasma_beta__',
                                                                                                        'omniweb__alfven_mach_number__',
                                                                                                        'omniweb__magnetosonic_mach_number__',
                                                                                                        'omniweb__s/cx_gse__[Re]',
                                                                                                        'omniweb__s/cy_gse__[Re]',
                                                                                                        'omniweb__s/cz_gse__[Re]'], nargs='+', type=str)
    parser.add_argument('--features_to_exclude_thermo', help='Features to exclude for thermospheric data', default=['all__dates_datetime__',
                                                                                                                    'tudelft_thermo__satellite__',
                                                                                                                    'tudelft_thermo__ground_truth_thermospheric_density__[kg/m**3]',
                                                                                                                    'NRLMSISE00__thermospheric_density__[kg/m**3]',
                                                                                                                    'JB08__thermospheric_density__[kg/m**3]'], nargs='+', type=str)
    parser.add_argument('--features_to_exclude_fism2_flare', help='Features to exclude for FISM2 flare data', default=['fism2_flare__uncertainty__',
                                                                                                                       'all__dates_datetime__'], nargs='+', type=str)
    parser.add_argument('--features_to_exclude_fism2_daily', help='Features to exclude for FISM2 daily data', default=['fism2_daily__uncertainty__',
                                                                                                                        'all__dates_datetime__'], nargs='+', type=str)
    parser.add_argument('--run_name', default='', help='Run name to be stored in wandb')


    opt = parser.parse_args()
    wandb.init(project='karman', config=vars(opt))
    if opt.run_name != '':
        wandb.run.name = opt.run_name
        wandb.run.save()

    print('Arguments:\n{}\n'.format(' '.join(sys.argv[1:])))
    print('Config:')
    pprint.pprint(vars(opt), depth=2, width=1)
    print()

    output_directory_exists = os.path.exists(opt.output_directory)

    if not output_directory_exists:
        # Create a new directory because it does not exist
        os.makedirs(opt.output_directory)
        print(f"Created directory for storing results: {opt.output_directory}")

    dataset=karman.ThermosphericDensityDataset(
        directory=opt.data_directory,
        exclude_omni=opt.exclude_omni,
        exclude_fism2_flare=opt.exclude_fism2_flare,
        exclude_fism2_daily=opt.exclude_fism2_daily,
        lag_minutes_omni=opt.lag_minutes_omni,
        lag_days_fism2_daily=opt.lag_days_fism2_daily,
        lag_minutes_fism2_flare=opt.lag_minutes_fism2_flare,
        wavelength_bands_to_skip=opt.wavelength_bands_to_skip,
        omniweb_downsampling_ratio=opt.omniweb_downsampling_ratio,
        features_to_exclude_omni=opt.features_to_exclude_omni,
        features_to_exclude_thermo=opt.features_to_exclude_thermo,
        features_to_exclude_fism2_flare=opt.features_to_exclude_fism2_flare,
        features_to_exclude_fism2_daily=opt.features_to_exclude_fism2_daily
    )

    #TODO Sort out how we deal with these indices. Storing the indices
    # is a bad way to go because the indices change based on the lag
    # Is there a way of generating them quickly? with parallelisaton...probably.
    print(f"Train, Valid, Test split:")
    if opt.load_indices==False:
        years = list(range(2003, 2022))
        months = np.array(range(1,13))
        train_idx = [0,1,2,5,6,8,9,10,11]
        validation_idx = [3,7]
        test_idx = [4]
        year_months = {}
        train_indices=[]
        val_indices=[]
        test_indices=[]
        for year in tqdm(years):
            year_months[year] = {}
            year_months[year]['train'] = np.roll(months, year)[train_idx]
            year_months[year]['validation'] = np.roll(months, year)[validation_idx]
            year_months[year]['test'] = np.roll(months, year)[test_idx]
            train_indices+=list(dataset.dates_thermo.index[(dataset.dates_thermo.dt.year.isin([year]) & dataset.dates_thermo.dt.month.isin(list(year_months[year]['train'])))].values)
            val_indices+=list(dataset.dates_thermo.index[(dataset.dates_thermo.dt.year.isin([year]) & dataset.dates_thermo.dt.month.isin(list(year_months[year]['validation'])))].values)
            test_indices+=list(dataset.dates_thermo.index[(dataset.dates_thermo.dt.year.isin([year]) & dataset.dates_thermo.dt.month.isin(list(year_months[year]['test'])))].values)
        print("Saving created indices to files:")
        with open(os.path.join(opt.data_directory, "train_indices.txt"), 'w') as output:
            for row in train_indices:
                output.write(str(row) + '\n')
        with open(os.path.join(opt.data_directory, "val_indices.txt"), 'w') as output:
            for row in val_indices:
                output.write(str(row) + '\n')
        with open(os.path.join(opt.data_directory, "test_indices.txt"), 'w') as output:
            for row in test_indices:
                output.write(str(row) + '\n')
    else:
        with open(os.path.join(opt.data_directory, "train_indices.txt"), 'r') as f:
            train_indices = [int(line.rstrip()) for line in f]
        with open(os.path.join(opt.data_directory, "val_indices.txt"), 'r') as f:
            val_indices = [int(line.rstrip()) for line in f]
        with open(os.path.join(opt.data_directory, "test_indices.txt"), 'r') as f:
            test_indices = [int(line.rstrip()) for line in f]

    print(f"Train set proportion: {round(len(train_indices)/len(dataset)*100, 2)} %, {len(train_indices)}/{len(dataset)}")
    print(f"Validation set proportion: {round(len(val_indices)/len(dataset)*100,2)} %, {len(val_indices)}/{len(dataset)}")
    print(f"Test set proportion: {round(len(test_indices)/len(dataset)*100, 2)} %, {len(test_indices)}/{len(dataset)}")
    print(f"Total dataset length: {len(dataset)}")

    train_indices=np.array(train_indices)
    val_indices=np.array(val_indices)
    test_indices=np.array(test_indices)

    #TODO fix issue with indices in datasets being too high...
    train_indices=train_indices[train_indices<len(dataset)]
    val_indices=val_indices[val_indices<len(dataset)]
    test_indices=test_indices[test_indices<len(dataset)]

    # Create train, valid, test dataloaders. Shuffle is false for validation and test
    # datasets to preserve order.
    train_loader = torch.utils.data.DataLoader(Subset(dataset, train_indices),
                                               batch_size=opt.batch_size,
                                               pin_memory=True,
                                               shuffle=True,
                                               num_workers=opt.num_workers,
                                               drop_last=True)
    valid_loader = torch.utils.data.DataLoader(Subset(dataset, val_indices),
                                               batch_size=opt.batch_size,
                                               pin_memory=True,
                                               num_workers=opt.num_workers,
                                               drop_last=True)
    test_loader = torch.utils.data.DataLoader(Subset(dataset, test_indices),
                                               batch_size=opt.batch_size,
                                               pin_memory=True,
                                               num_workers=opt.num_workers,
                                               drop_last=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if opt.model == 'FeedForwardDensityPredictor':
        # Will only use an FFNN with just the thermo static features data
        model = FeedForwardDensityPredictor(
                            num_features=dataset.data_thermo_matrix.shape[1]
                            )
    elif opt.model=='Fism2FlareDensityPredictor':
        if opt.exclude_fism2_daily and opt.exclude_omni:
            model=Fism2FlareDensityPredictor(
                            input_size_thermo=dataset.data_thermo_matrix.shape[1],
                            input_size_fism2_flare=dataset.fism2_flare_irradiance_matrix.shape[1],
                            output_size_fism2_flare=20
                            )
        else:
            raise RuntimeError(f"exclude_fism2_daily and exclude_omni are not set to True; while model chosen is {opt.model}")
    elif opt.model=='Fism2DailyDensityPredictor':
        if opt.exclude_fism2_flare and opt.exclude_omni:
            model=Fism2DailyDensityPredictor(
                            input_size_thermo=dataset.data_thermo_matrix.shape[1],
                            input_size_fism2_daily=dataset.fism2_daily_irradiance_matrix.shape[1],
                            output_size_fism2_daily=20
                            )
        else:
            raise RuntimeError(f"exclude_fism2_flare and exclude_omni are not set to True; while model chosen is {opt.model}")
    elif opt.model=='OmniDensityPredictor':
        if opt.exclude_fism2_daily and opt.exclude_fism2_flare:
            model=OmniDensityPredictor(
                            input_size_thermo=dataset.data_thermo_matrix.shape[1],
                            input_size_omni=dataset.data_omni_matrix.shape[1],
                            output_size_omni=20
                            )
        else:
            raise RuntimeError(f"exclude_fism2_daily and exclude_fism2_flare are not set to True; while model chosen is {opt.model}")
    elif opt.model == 'FullFeatureDensityPredictor':
        if opt.exclude_omni==False and opt.exclude_fism2_flare==False and opt.exclude_fism2_daily==False:
            model = FullFeatureDensityPredictor(
                            input_size_thermo=dataset.data_thermo_matrix.shape[1],
                            input_size_fism2_flare=dataset.fism2_flare_irradiance_matrix.shape[1],
                            input_size_fism2_daily=dataset.fism2_daily_irradiance_matrix.shape[1],
                            input_size_omni=dataset.data_omni_matrix.shape[1],
                            output_size_fism2_flare=20,
                            output_size_fism2_daily=20,
                            output_size_omni=20
                            )
        else:
            raise RuntimeError("exclude_omni, exclude_fism2_daily, exclude_fism2_flare are not all True, FullFeatureDensityPredictor cannot be used")
    if torch.cuda.device_count()>1:
        print(f"Parallelizing the model on {torch.cuda.device_count()} GPUs")
        model=nn.DataParallel(model)
    print(f"Device is: {device}")
    model.to(device)
    if opt.optimizer=='adam':
        optimizer=optim.Adam(model.parameters(), lr=opt.learning_rate, weight_decay=opt.weight_decay)
    elif opt.optimizer=='sgd':
        optimizer=optim.SGD(model.parameters(), lr=opt.learning_rate, weight_decay=opt.weight_decay)

    time_start=datetime.datetime.now()

    best_model_path=os.path.join(opt.output_directory,"best_model_"+opt.model+f"_{time_start}")
    last_model_path=os.path.join(opt.output_directory,"last_model_"+opt.model+f"_{time_start}")

    best_validation_loss = np.inf
    for epoch in range(opt.epochs):
        if epoch%opt.valid_every==0:
            print("Validation\n")
            #model.eval()
            model.train(False)
            batches_valid_loss=0
            validation_losses=[]
            with torch.no_grad():
                for batch_val in tqdm(valid_loader):
                    [batch_val.__setitem__(key, batch_val[key].to(device)) for key in batch_val.keys()]
                    output_val = model(batch_val)
                    validation_loss = nn.MSELoss()(output_val, batch_val['target'].unsqueeze(1))
                    validation_losses.append(float(validation_loss))
            epoch_validation_loss =  np.mean(validation_losses)
            wandb.log({'validation_loss': epoch_validation_loss})

            if epoch_validation_loss < best_validation_loss:
                best_validation_loss = epoch_validation_loss
                #TODO this is very similar to simply evaluating on the
                # validation set. Can probably be combined
                print(f"Saving best model to: {best_model_path} \n")
                torch.save({'state_dict': model.state_dict(),
                            'opt': opt}, best_model_path)
#torch.save(model.state_dict(), best_model_path)

            model.train(True)
        for batch in tqdm(train_loader):
            #TODO: this will be modified once we will be able to handle lags in the NN part
            #send all batch elements to device
            [batch.__setitem__(key, batch[key].to(device)) for key in batch.keys()]
            optimizer.zero_grad()
            output = model(batch)
            train_loss=nn.MSELoss()(output, batch['target'].unsqueeze(1))
            train_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            print((epoch, float(train_loss)),end='\r')
            wandb.log({'train_loss': train_loss.item()})
            #i_total+=1
        if epoch%opt.test_every==0:
            print("Testing\n")
            #model.eval()
            model.train(False)
            batches_test_loss=0
            test_losses=[]
            with torch.no_grad():
                for batch_test in tqdm(test_loader):
                    [batch_test.__setitem__(key, batch_test[key].to(device)) for key in batch_test.keys()]
                    output_test = model(batch_test)
                    test_loss = nn.MSELoss()(output_test, batch_test['target'].unsqueeze(1))
                    test_losses.append(float(test_loss))
            epoch_test_loss =  np.mean(test_losses)
            wandb.log({'test_loss': epoch_test_loss})

    print("Benchmarking on the test set (using the best model in terms of validation loss):")
    print(f"Saving last model to: {last_model_path}\n")
    torch.save({'state_dict':model.state_dict(),
                'opt':opt},last_model_path)
    model.load_state_dict(torch.load(best_model_path)['state_dict'])
    test_results = karman.Benchmark(
        batch_size=opt.batch_size,
        num_workers=opt.num_workers,
        data_directory=opt.data_directory
    ).evaluate_model(dataset, model)
    wandb.log({
        'Test Results': test_results
    })
#            if opt.test_mode and i % 5 == 0:
                # Quickly test whether script is working on a much
                # smaller train iteration
#                break


if __name__ == "__main__":
    time_start = time.time()
    run()
    print('\nTotal duration: {}'.format(time.time() - time_start))
    sys.exit(0)

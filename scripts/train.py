import argparse
import os
import pprint
import sys
import time

import datetime
import karman
from karman.nn import SimpleNN, AddOmni
import numpy as np
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset
from tqdm import tqdm
import wandb
from pyfiglet import Figlet
from termcolor import colored
import copy
from torch.utils.data import RandomSampler
import pandas as pd
import random
torch.multiprocessing.set_sharing_strategy('file_system')

def validate_model(model, loader, loss_function, device):
    losses = []
    with torch.no_grad():
        for batch in tqdm(loader):
            [batch.__setitem__(key, batch[key].to(dtype=torch.float32).to(device)) for key in batch.keys()]
            output = model(batch)
            loss = loss_function(output, batch['target'].unsqueeze(1))
            losses.append(float(loss))
    return np.mean(losses)

def run():
    print('Karman Model Training')
    f = Figlet(font='5lineoblique')
    print(colored(f.renderText('KARMAN'), 'red'))
    f = Figlet(font='digital')
    print(colored(f.renderText("Thermospheric  density  calculations"), 'blue'))
    print(colored(f'Version {karman.__version__}\n','blue'))

    parser = argparse.ArgumentParser(description='Karman', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--output_directory', help='Output directory', default='output_directory',required=True)
    parser.add_argument('--epochs', '-n', help='Number of epochs', default=10, type=int)
    parser.add_argument('--epochs_per_validation', default=1, type=int)
    parser.add_argument('--data_directory', default='/home/jupyter/karman-project/data_directory', type=str)
    parser.add_argument('--learning_rate', help='learning rate to use', default=1e-4, type=float)
    parser.add_argument('--weight_decay', help='Weight decay: optimizer parameter', default=0., type=float)
    parser.add_argument('--optimizer', help='Optimizer to use', default='adam', choices=['adam', 'sgd'])
    parser.add_argument('--lag_minutes_omni', help='Time lag (in minutes) to consider for the OMNIWeb data', default=6*60, type=float)
    parser.add_argument('--features_to_exclude_omni', help='Features to exclude for OMNI data', default='all__dates_datetime__,omniweb__id_for_imf_spacecraft__,omniweb__id_for_sw_plasma_spacecraft__,omniweb__#_of_points_in_imf_averages__,omniweb__#_of_points_in_plasma_averages__,omniweb__percent_of_interpolation__,omniweb__timeshift__[s],omniweb__rms_timeshift__[s],omniweb__rms_min_variance__[s**2],omniweb__time_btwn_observations__[s],omniweb__rms_sd_b_scalar__[nT],omniweb__rms_sd_b_field_vector__[nT],omniweb__flow_pressure__[nPa],omniweb__electric_field__[mV/m],omniweb__plasma_beta__,omniweb__alfven_mach_number__,omniweb__magnetosonic_mach_number__,omniweb__s/cx_gse__[Re],omniweb__s/cy_gse__[Re],omniweb__s/cz_gse__[Re]', type=str)
    parser.add_argument('--features_to_exclude_thermo', help='Features to exclude for thermospheric data', default='all__dates_datetime__,tudelft_thermo__satellite__,tudelft_thermo__ground_truth_thermospheric_density__[kg/m**3],NRLMSISE00__thermospheric_density__[kg/m**3],JB08__thermospheric_density__[kg/m**3]', type=str)
    parser.add_argument('--features_to_exclude_fism2_flare_stan_bands',
                        help='Features to exclude for fism2 flare stan bands',
                        default='all__dates_datetime__',
                        type=str)
    parser.add_argument('--features_to_exclude_fism2_daily_stan_bands',
                        help='Features to exclude for fism2 daily stan bands',
                        default='all__dates_datetime__',
                        type=str)
    parser.add_argument('--omni_resolution',
                        help='Resolution in minutes for omniweb geomagnetic data',
                        default=30, type=int)
    parser.add_argument('--fism2_flare_stan_bands_resolution',
                        help='Resolution in minutes for Fism2 flare stan ban data',
                        default=30, type=int)
    parser.add_argument('--fism2_daily_stan_bands_resolution',
                        help='Resolution in minutes for Fism2 daily stan ban data (1440 per day)',
                        default=1440, type=int)
    parser.add_argument('--lag_fism2_minutes_flare_stan_bands', default=6*60, type=int)
    parser.add_argument('--lag_fism2_minutes_daily_stan_bands', default=1440, type=int)
    parser.add_argument('--include_omni', default=False, type=bool)
    parser.add_argument('--include_daily_stan_bands', default=False, type=bool)
    parser.add_argument('--include_flare_stan_bands', default=False, type=bool)
    parser.add_argument('--run_name', default='', help='Run name to be stored in wandb')
    parser.add_argument('--cyclical_features', default=True, type=bool)
    parser.add_argument('--model',
                        default='FullFeatureFeedForward',
                        choices=['AddOmni',
                                 'SimpleNN'
                                 ])
    parser.add_argument('--dropout', default=0.0, type=float)
    parser.add_argument('--folds',
                        default='1',
                        help='Split by comma e.g. 1,2,3,4,5')
    parser.add_argument('--hidden_size', default=200, type=int)
    parser.add_argument('--out_features', default=50, type=int)
    parser.add_argument('--train_subsample', default=None)
    parser.add_argument('--run_benchmark', default=True, type=bool)
    parser.add_argument('--run_tests', default=True, type=bool)
    parser.add_argument('--max_altitude', default=600_000, type=float)
    parser.add_argument('--lstm_depth', default=2, type=int)
    parser.add_argument('--seeds', default=1, type=int)

    opt = parser.parse_args()
    wandb.init(project='karman-project', config=vars(opt), entity='eddyb92')
    # wandb.init(mode="disabled")
    if opt.run_name != '':
        wandb.run.name = opt.run_name
        wandb.run.save()

    print('Arguments:\n{}\n'.format(' '.join(sys.argv[1:])))
    print('Config:')
    pprint.pprint(vars(opt), depth=2, width=1)
    print()

    if not os.path.exists(opt.output_directory):
        # Create a new directory because it does not exist
        os.makedirs(opt.output_directory)
        print(f"Created directory for storing results: {opt.output_directory}")

    dataset=karman.ThermosphericDensityDataset(
        directory=opt.data_directory,
        lag_minutes_omni=opt.lag_minutes_omni,
        lag_minutes_fism2_flare_stan_bands=opt.lag_fism2_minutes_flare_stan_bands,
        lag_minutes_fism2_daily_stan_bands=opt.lag_fism2_minutes_daily_stan_bands,
        omni_resolution=opt.omni_resolution,
        fism2_flare_stan_bands_resolution=opt.fism2_flare_stan_bands_resolution,
        fism2_daily_stan_bands_resolution=opt.fism2_daily_stan_bands_resolution,
        features_to_exclude_thermo=opt.features_to_exclude_thermo.split(','),
        features_to_exclude_omni=opt.features_to_exclude_omni.split(','),
        features_to_exclude_fism2_flare_stan_bands=opt.features_to_exclude_fism2_flare_stan_bands.split(','),
        features_to_exclude_fism2_daily_stan_bands=opt.features_to_exclude_fism2_daily_stan_bands.split(','),
        include_omni=opt.include_omni,
        include_daily_stan_bands=opt.include_daily_stan_bands,
        include_flare_stan_bands=opt.include_flare_stan_bands,
        create_cyclical_features=opt.cyclical_features,
        max_altitude=opt.max_altitude
    )

    benchmark_results = []
    test_seed_losses = []

    for fold in opt.folds.split(','):
        print(f"Running fold {fold}")
        benchmark_results = []
        for seed in range(opt.seeds):
            print(f"Running seed: {seed}")
            np.random.seed(seed)
            torch.manual_seed(seed)
            random.seed(seed)
            if opt.model == 'SimpleNN':
                model = SimpleNN(
                    hidden_size=opt.hidden_size,
                    out_features=opt.out_features).to(dtype=torch.float32)
            elif opt.model == 'AddOmni':
                model = AddOmni(
                    hidden_size=opt.hidden_size,
                    out_features=opt.out_features).to(dtype=torch.float32)

            validation_step_loader = torch.utils.data.DataLoader(dataset,
                                                                batch_size=2,
                                                                num_workers=0)
            #Need to do this if using the LazyLinear Module (avoids having to hard code input layers to a linear layer)..sue me
            model.forward(next(iter(validation_step_loader)))

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            print(f"Device is: {device}")
            model.to(device).to(dtype=torch.float32)

            optimizer=optim.Adam(model.parameters(), lr=opt.learning_rate, weight_decay=opt.weight_decay)

            test_month_idx = 2 * (int(fold) - 1)
            validation_month_idx = test_month_idx + 2
            dataset._set_indices(test_month_idx=[test_month_idx], validation_month_idx=[validation_month_idx])
            train_dataset = dataset.train_dataset()
            if opt.train_subsample is not None:
                train_sampler = RandomSampler(train_dataset, num_samples=int(opt.train_subsample))
            else:
                train_sampler = RandomSampler(train_dataset, num_samples=len(train_dataset))
            train_loader = torch.utils.data.DataLoader(train_dataset,
                                        batch_size=opt.batch_size,
                                        pin_memory=True,
                                        num_workers=opt.num_workers,
                                        sampler=train_sampler,
                                        drop_last=True)
            validation_loader = torch.utils.data.DataLoader(dataset.validation_dataset(),
                                                    batch_size=opt.batch_size,
                                                    pin_memory=False,
                                                    num_workers=opt.num_workers,
                                                    drop_last=False)
            test_loader = torch.utils.data.DataLoader(dataset.test_dataset(),
                                            batch_size=opt.batch_size,
                                            pin_memory=False,
                                            num_workers=opt.num_workers,
                                            drop_last=False)
            best_model_path=os.path.join(opt.output_directory,"best_model_"+opt.model+f"_{time_start}_fold_{fold}_seed_{seed}")
            loss_function = nn.MSELoss()
            best_validation_loss = np.inf
            samples_trained = 0
            for epoch in range(opt.epochs):
                model.train(True)
                for i, batch in tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Running Train epoch {epoch}'):
                    [batch.__setitem__(key, batch[key].to(device).to(torch.float32)) for key in batch.keys()]
                    optimizer.zero_grad()
                    output = model(batch)
                    train_loss = loss_function(output, batch['target'].unsqueeze(1))
                    train_loss.backward()
                    optimizer.step()
                    wandb.log({f'train_loss_fold_{fold}_seed_{seed}': float(train_loss.detach().item())})
                    #So we can compare plots of different batch sizes
                    samples_trained += int(opt.batch_size)
                    wandb.log({'samples_trained': samples_trained})

                if epoch % opt.epochs_per_validation == 0:
                    print("Validating\n")
                    validation_loss = validate_model(model, validation_loader, loss_function, device)
                    wandb.log({f'validation_loss_fold_{fold}_seed_{seed}': validation_loss})
                    if opt.run_tests:
                        print("Testing\n")
                        test_loss = validate_model(model, test_loader, loss_function, device)
                        wandb.log({f'test_loss_fold_{fold}_seed_{seed}': test_loss})

                    if validation_loss < best_validation_loss:
                        best_validation_loss = validation_loss
                        wandb.log({f'best_validation_loss_fold_{fold}_seed_{seed}': best_validation_loss})
                        reported_test_loss = test_loss
                        print(f"Saving best model to: {best_model_path} \n")
                        torch.save({'state_dict': model.state_dict(),
                                    'opt': opt}, best_model_path)
                        wandb.log({f'reported_test_loss_fold_{fold}_seed_{seed}': test_loss})
                        best_state_dict = copy.deepcopy(model.state_dict())

            test_seed_losses.append(reported_test_loss)

            # Run benchmark at the end of fold
            if opt.run_benchmark:
                print('Running benchmarks')
                model.load_state_dict(best_state_dict)
                seed_model_name = f'{opt.run_name}_fold_{fold}_seed_{seed}'
                seed_benchmark_results = Benchmark(batch_size=opt.batch_size,
                                            num_workers=opt.num_workers,
                                            data_directory=opt.data_directory,
                                            output_directory=opt.output_directory,
                                            model_name=seed_model_name).evaluate_model(dataset, model)

                for row in seed_benchmark_results.iterrows():
                    if row[1]['Model'] == seed_model_name:
                        wandb.log({f"reported_test_fold_{fold}_seed_{seed}_{row[1]['Metric Type']}_{row[1]['Condition']}": row[1]['Metric Value']})
                #Ignore NRLMSISE and JB08 entries
                benchmark_results.append(seed_benchmark_results[seed_benchmark_results['Model'] == seed_model_name])

        benchmark_results = pd.concat(benchmark_results, ignore_index=True)
        #Skipna is True because sometimes the categories might be empty, like in severe storms.
        benchmark_results_mean = benchmark_results.dropna().groupby(['Metric Type', 'Condition']).mean()
        benchmark_results_std = benchmark_results.dropna().groupby(['Metric Type', 'Condition']).std()

        for row in benchmark_results_mean.iterrows():
            wandb.log({f"reported_test_fold_{fold}_mean_{row[0][0]}_{row[0][1]}": row[1]['Metric Value']})
        for row in benchmark_results_std.iterrows():
            wandb.log({f"reported_test_fold_{fold}_std_{row[0][0]}_{row[0][1]}": row[1]['Metric Value']})

        wandb.log({f'reported_test_loss_fold_{fold}_mean': np.mean(test_seed_losses)})
        wandb.log({f'reported_test_loss_fold_{fold}_std': np.std(test_seed_losses)})


if __name__ == "__main__":
    time_start = time.time()
    run()
    print('\nTotal duration: {}'.format(time.time() - time_start))
    sys.exit(0)

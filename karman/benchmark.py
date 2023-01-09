
import pandas as pd
import torch
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import numpy as np
import karman
from karman.nn import *
import os
from torch.utils.data import Subset
from torch import nn
import argparse
from pyfiglet import Figlet
from termcolor import colored

class Benchmark():
    def __init__(self,
                 batch_size=512,
                 num_workers=30,
                 output_directory='output_directory',
                 data_directory='/home/jupyter/data',
                 model_name='model_to_benchmark'):
        self.output_directory = output_directory
        self.model_name = model_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_directory = data_directory
        self.jb08_column = 'JB08__thermospheric_density__[kg/m**3]'
        self.nrlmsise_column = 'NRLMSISE00__thermospheric_density__[kg/m**3]'
        self.density_column = 'tudelft_thermo__ground_truth_thermospheric_density__[kg/m**3]'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.density_threshold_ignore = 1e-17
        self.metrics = [
            rmse,
            mape,
            correlation
        ]
        self.ap_column = 'celestrack__ap_h_0__'
        # Numbers are put in to preserve alphabetical order.

        self.storm_thresholds = {
            '1. (0-15) Quiet': 15.0,
            '2. (15-30) Mild': 30.0,
            '3. (30-50) Minor': 50.0,
            '4. (50+) Major' : 400.0,
        }
        self.altitude_column = 'tudelft_thermo__altitude__[m]'
        self.altitude_ranges = {
            '200km-250km': 250_000,
            '250km-300km': 300_000,
            '300km-350km': 350_000,
            '350km-400km': 400_000,
            '400km-450km': 450_000,
            '450km-500km': 500_000,
            '500km-550km': 550_000,
        }
        self.metrics = {
            'RMSE': rmse,
            'MAPE': mape,
        }

        self.model_column_map = {
            'JB08': 'JB08__thermospheric_density__[kg/m**3]',
            'Nrlmsise': 'NRLMSISE00__thermospheric_density__[kg/m**3]',
            self.model_name: 'model'
        }
        # Models use dropout and therefore have stochastic outputs. This
        # means that we need to run multiple runs to get the distribution
        self.model_runs = 1

    def evaluate_model(self, dataset, model):
        """
        Evaluates the model on the dataset with multiple metrics
        """
        test_dataset = self.get_predictions_and_targets(dataset, model)
        return self.analyze_dataframe(test_dataset)

    def analyze_dataframe(self, test_dataset):
        print('Evaluating Storm Condition Results.')
        results = pd.DataFrame(columns=['Model', 'Metric Value', 'Condition', 'Support', 'Metric Type'])
        storm_bins = np.digitize(
            test_dataset[self.ap_column].astype(float).values,
            np.array(list(self.storm_thresholds.values()))
        )
        test_dataset['storm_classification'] = storm_bins

        for storm_index, storm_condition in enumerate(self.storm_thresholds.keys()):
            storm_dataset = test_dataset[test_dataset['storm_classification'] == storm_index]
            target = storm_dataset[self.density_column].values.astype(float)
            for model, column_name in self.model_column_map.items():
                for metric, metric_function in self.metrics.items():
                    prediction = storm_dataset[column_name].values.astype(float)
                    entry = {
                        'Model': model,
                        'Metric Type': metric,
                        'Condition': storm_condition,
                        'Metric Value': metric_function(prediction, target),
                        'Support': len(storm_dataset),
                    }
                    results = results.append(entry, ignore_index=True)

        altitude_bins = np.digitize(
            test_dataset[self.altitude_column].astype(float).values,
            np.array(list(self.altitude_ranges.values()))
        )
        test_dataset['altitude_classification'] = altitude_bins

        for altitude_index, altitude_range in enumerate(self.altitude_ranges.keys()):
            altitude_dataset = test_dataset[test_dataset['altitude_classification'] == altitude_index]
            target = altitude_dataset[self.density_column].values.astype(float)
            for model, column_name in self.model_column_map.items():
                for metric, metric_function in self.metrics.items():
                    prediction = altitude_dataset[column_name].values.astype(float)
                    entry = {
                        'Model': model,
                        'Metric Type': metric,
                        'Condition': altitude_range,
                        'Metric Value': metric_function(prediction, target),
                        'Support': len(altitude_dataset),
                    }
                    results = results.append(entry, ignore_index=True)

        target = test_dataset[self.density_column].values.astype(float)
        for model, column_name in self.model_column_map.items():
            for metric, metric_function in self.metrics.items():
                prediction = test_dataset[column_name].values.astype(float)
                entry = {
                    'Model': model,
                    'Metric Type': metric,
                    'Condition': 'No condition',
                    'Metric Value': metric_function(prediction, target),
                    'Support': len(test_dataset),
                }
                results = results.append(entry, ignore_index=True)

        output_file = os.path.join(self.output_directory, f'{self.model_name}_results.csv')
        print('Saving results to', output_file)
        results.to_csv(output_file)
        return results

    def get_predictions_and_targets(self, dataset, model):
        """
        Gather up the predictions and targets from the dataset.
        """
        targets = []
        predictions = []

        loader = DataLoader(dataset.test_dataset(),
                            batch_size=self.batch_size,
                            num_workers=self.num_workers,
                            drop_last=False)

        with torch.no_grad():
            # Dropout seems to work awfully with regression. However, in the case that it is enabled
            # we can run the model multiple times to take a mean.
            model.train(True)
            for batch in tqdm(loader):
                [batch.__setitem__(key, batch[key].to(self.device)) for key in batch.keys()]
                if model.dropout == 0.0:
                    times_to_run = 1
                else:
                    times_to_run = self.model_runs
                dropout_predictions = []
                for _ in range(times_to_run):
                    dropout_predictions.append(model.forward(batch))
                dropout_predictions = torch.stack(dropout_predictions, dim=1)
                targets.append(batch['target'])
                predictions.append(torch.mean(dropout_predictions, dim=1))

        predictions = torch.flatten(
            torch.cat(predictions)
        ).detach().cpu().numpy()

        targets = torch.flatten(
            torch.cat(targets)
        ).detach().cpu().numpy()

        predictions = dataset.unscale_density(predictions)
        targets = dataset.unscale_density(targets)
        dataset.data_thermo['data']['model'] = 0.0
        dataset.data_thermo['data'].loc[dataset.test_indices, 'model'] = predictions.astype(float)
        test_dataset = dataset.data_thermo['data'].loc[dataset.test_indices, :]
        # Only save columns that are useful (its a lot of data)
        test_dataset_columns = [
            self.jb08_column,
            self.nrlmsise_column,
            self.density_column,
            self.ap_column,
            self.altitude_column,
            'model',
        ]
        test_dataset = test_dataset[test_dataset_columns]
        # Ignore these densities treated as outliers
        print('Ignoring outliers...')
        starting_length = len(test_dataset)
        test_dataset = test_dataset[test_dataset[self.density_column] > self.density_threshold_ignore]
        end_length = len(test_dataset)
        print(f'Removed {starting_length - end_length} entries with outlier densities below {self.density_threshold_ignore}')
        return test_dataset


def mape(prediction, target):
    mape_ = 100 * np.mean(np.divide(
        np.abs(target - prediction),
        target
    ))
    return mape_

def mse(x, y):
    z = x - y
    return np.mean(z*z)

def rmse(x, y):
    return np.sqrt(mse(x, y))

def correlation(x,y):
    return np.corrcoef(x,y)[0,1]

if __name__ == '__main__':
    print('Karman Model Evaluation')
    f = Figlet(font='5lineoblique')
    print(colored(f.renderText('KARMAN'), 'red'))
    f = Figlet(font='digital')
    print(colored(f.renderText("Thermospheric  density  calculations"), 'blue'))
    print(colored(f'Version {karman.__version__}\n','blue'))

    parser = argparse.ArgumentParser(description='Karman', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', help='Random number seed', default=1, type=int)
    parser.add_argument('--model_name', help='Descriptive name of model', default='model', type=str)
    parser.add_argument('--model_folder', help='Where is the model stored?', type=str)
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--num_workers', default=30, type=int)
    parser.add_argument('--data_directory', default='/home/jupyter/karman-project/data_directory', type=str)

    opt = parser.parse_args()

    #Stick to this convention 'best_model' is the model of interest.
    model_path = [os.path.join(opt.model_folder, f) for f in os.listdir(opt.model_folder) if 'best_model' in f][0]

    print('Loading Data')
    model_opt=torch.load(model_path)['opt']
    print(model_opt)
    # import pdb; pdb.set_trace()

    dataset=karman.ThermosphericDensityDataset(
        directory=opt.data_directory,
        # lag_minutes_omni=model_opt.lag_minutes_omni,
        # lag_minutes_fism2_flare_stan_bands=model_opt.lag_fism2_minutes_flare_stan_bands,
        # lag_minutes_fism2_daily_stan_bands=model_opt.lag_fism2_minutes_daily_stan_bands,
        # omni_resolution=model_opt.omniweb_downsampling_ratio,
        # fism2_flare_stan_bands_resolution=model_opt.fism2_flare_stan_bands_resolution,
        # fism2_daily_stan_bands_resolution=model_opt.fism2_daily_stan_bands_resolution,
        features_to_exclude_thermo=model_opt.features_to_exclude_thermo,
        # features_to_exclude_omni=model_opt.features_to_exclude_omni.split(','),
        # features_to_exclude_fism2_flare_stan_bands=model_opt.features_to_exclude_fism2_flare_stan_bands.split(','),
        # features_to_exclude_fism2_daily_stan_bands=model_opt.features_to_exclude_fism2_daily_stan_bands.split(','),
        create_cyclical_features=True,
        # max_altitude=model_opt.max_altitude,
    )

    print('Loading Model')

    if model_opt.model == 'FullFeatureFeedForward':
        model = FullFeatureFeedForward(dropout=model_opt.dropout).to(dtype=torch.float32)

    state_dict = torch.load(os.path.join(model_path))['state_dict']
    #Sanitize state_dict key names
    for key in list(state_dict.keys()):
        if key.startswith('module'):
        # Model was saved as dataparallel model
            # Remove 'module.' from start of key
            state_dict[key[7:]] = state_dict.pop(key)
        else:
            continue
    model.load_state_dict(state_dict)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #Need to do this if using the LazyLinear Module (avoids having to hard code input layers to a linear layer)
    dummy_loader = torch.utils.data.DataLoader(dataset.validation_dataset(),
                                                        batch_size=2,
                                                        pin_memory=False,
                                                        num_workers=0,
                                                        drop_last=True)
    model.forward(next(iter(dummy_loader)))
    model.to(device)

    benchmark = Benchmark(
        batch_size=opt.batch_size,
        num_workers=opt.num_workers,
        data_directory=model_opt.data_directory,
        output_directory=opt.model_folder,
        model_name=opt.model_name).evaluate_model(dataset, model)

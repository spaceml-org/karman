
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
        with open(os.path.join(self.data_directory, "test_indices.txt"), 'r') as f:
            self.test_indices = [int(line.rstrip()) for line in f]
        self.jb08_column = 'JB08__thermospheric_density__[kg/m**3]'
        self.nrlmsise_column = 'NRLMSISE00__thermospheric_density__[kg/m**3]'
        self.density_column = 'tudelft_thermo__ground_truth_thermospheric_density__[kg/m**3]'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.density_threshold_ignore = 1e-17
        self.metrics = [
            rmse,
            mpe,
            correlation
        ]
        self.ap_column = 'celestrack__ap_h_0__'
        self.storm_thresholds = {
            'Quiet': 15.0,
            'Mild': 30.0,
            'Minor': 50.0,
            'Major' : 100.0,
            'Severe': 400.0
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
            'MPE': mpe,
            'Correlation': correlation
        }

        self.model_column_map = {
            'JB08': 'JB08__thermospheric_density__[kg/m**3]',
            'Nrlmsise': 'NRLMSISE00__thermospheric_density__[kg/m**3]',
            'Model': 'model'
        }

    def evaluate_model(self, dataset, model):
        """
        Evaluates the model on the dataset with multiple metrics
        """
        test_dataset = self.get_predictions_and_targets(dataset, model)
        self.analyize_dataframe(test_dataset)

    def analyize_dataframe(self, test_dataset):
        print('Evaluating Storm Condition Results.')
        storm_bins = np.digitize(
            test_dataset[self.ap_column].astype(float).values,
            np.array(list(self.storm_thresholds.values()))
        )
        test_dataset['storm_classification'] = storm_bins
        storm_results = pd.DataFrame(columns=['Metric Value', 'Condition', 'Support', 'Metric Type'])

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
                    storm_results = storm_results.append(entry, ignore_index=True)

        storm_file = os.path.join(self.output_directory, f'{self.model_name}_storm_results.csv')
        print('Saving storm results to', storm_file)
        storm_results.to_csv(storm_file)

        print('Evaluating Altitude Results.')
        altitude_bins = np.digitize(
            test_dataset[self.altitude_column].astype(float).values,
            np.array(list(self.altitude_ranges.values()))
        )
        test_dataset['altitude_classification'] = altitude_bins
        altitude_results = pd.DataFrame(columns=['Metric Value', 'Condition', 'Support', 'Metric Type'])

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
                    altitude_results = altitude_results.append(entry, ignore_index=True)

        altitude_file = os.path.join(self.output_directory, f'{self.model_name}_altitude_results.csv')
        print('Saving storm results to', altitude_file)
        altitude_results.to_csv(os.path.join(self.output_directory, f'{self.model_name}_altitude_results.csv'))

    def get_predictions_and_targets(self, dataset, model):
        """
        Gather up the predictions and targets from the dataset.
        """
        targets = []
        predictions = []
        test_indices=np.array(self.test_indices)
        test_indices=test_indices[test_indices<len(dataset)]
        dataset_test_indices = [dataset.index_list[i] for i in test_indices]

        loader = DataLoader(Subset(dataset, test_indices),
                            batch_size=self.batch_size,
                            num_workers=self.num_workers)

        with torch.no_grad():
            model.train(False)
            for batch in tqdm(loader):
                [batch.__setitem__(key, batch[key].to(self.device)) for key in batch.keys()]
                prediction = model.forward(batch)
                targets.append(batch['target'])
                predictions.append(prediction)

        predictions = torch.flatten(
            torch.cat(predictions)
        ).detach().cpu().numpy()

        targets = torch.flatten(
            torch.cat(targets)
        ).detach().cpu().numpy()

        predictions = dataset.unscale_density(predictions)
        targets = dataset.unscale_density(targets)

        dataset.data_thermo['model'] = 0.0
        dataset.data_thermo.loc[dataset_test_indices, 'model'] = predictions.astype(float)
        test_dataset = dataset.data_thermo.loc[dataset_test_indices, :]
        model_results_file = os.path.join(self.output_directory, f'{self.model_name}_model_outputs.csv')
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
        print('Saving model outputs to', model_results_file)
        test_dataset.to_csv(model_results_file)
        return test_dataset


def mpe(prediction, target):
    mpe_ = 100 * np.mean(np.divide(
        np.abs(target - prediction),
        target
    ))
    return mpe_

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

    dataset = karman.ThermosphericDensityDataset(
        directory=opt.data_directory,
        exclude_omni=model_opt.exclude_omni,
        exclude_fism2_daily=model_opt.exclude_fism2_daily,
        exclude_fism2_flare=model_opt.exclude_fism2_flare,
        lag_minutes_omni=model_opt.lag_minutes_omni,
        lag_days_fism2_daily=model_opt.lag_days_fism2_daily,
        lag_minutes_fism2_flare=model_opt.lag_minutes_fism2_flare,
        wavelength_bands_to_skip=model_opt.wavelength_bands_to_skip,
        omniweb_downsampling_ratio=model_opt.omniweb_downsampling_ratio,
        features_to_exclude_omni=model_opt.features_to_exclude_omni,
        features_to_exclude_thermo=model_opt.features_to_exclude_thermo,
        features_to_exclude_fism2_flare=model_opt.features_to_exclude_fism2_flare,
        features_to_exclude_fism2_daily=model_opt.features_to_exclude_fism2_daily
    )

    print('Loading Model')

    if model_opt.model == 'FeedForwardDensityPredictor':
        # Will only use an FFNN with just the thermo static features data
        model = FeedForwardDensityPredictor(
                            num_features=dataset.data_thermo_matrix.shape[1]
                            )
    elif model_opt.model=='Fism2FlareDensityPredictor':
        if model_opt.exclude_fism2_daily and model_opt.exclude_omni:
            model=Fism2FlareDensityPredictor(
                            input_size_thermo=dataset.data_thermo_matrix.shape[1],
                            input_size_fism2_flare=dataset.fism2_flare_irradiance_matrix.shape[1],
                            output_size_fism2_flare=20
                            )
        else:
            raise RuntimeError(f"exclude_fism2_daily and exclude_omni are not set to True; while model chosen is {model_opt.model}")
    elif model_opt.model=='Fism2DailyDensityPredictor':
        if model_opt.exclude_fism2_flare and model_opt.exclude_omni:
            model=Fism2DailyDensityPredictor(
                            input_size_thermo=dataset.data_thermo_matrix.shape[1],
                            input_size_fism2_daily=dataset.fism2_daily_irradiance_matrix.shape[1],
                            output_size_fism2_daily=20
                            )
        else:
            raise RuntimeError(f"exclude_fism2_flare and exclude_omni are not set to True; while model chosen is {model_opt.model}")
    elif model_opt.model=='OmniDensityPredictor':
        if model_opt.exclude_fism2_daily and model_opt.exclude_fism2_flare:
            model=OmniDensityPredictor(
                            input_size_thermo=dataset.data_thermo_matrix.shape[1],
                            input_size_omni=dataset.data_omni_matrix.shape[1],
                            output_size_omni=20
                            )
        else:
            raise RuntimeError(f"exclude_fism2_daily and exclude_fism2_flare are not set to True; while model chosen is {model_opt.model}")
    elif model_opt.model == 'FullFeatureDensityPredictor':
        if model_opt.exclude_omni==False and model_opt.exclude_fism2_flare==False and model_opt.exclude_fism2_daily==False:
            model = FullFeatureDensityPredictor(
                            input_size_thermo=dataset.data_thermo_matrix.shape[1],
                            input_size_fism2_flare=dataset.fism2_flare_irradiance_matrix.shape[1],
                            input_size_fism2_daily=dataset.fism2_daily_irradiance_matrix.shape[1],
                            input_size_omni=dataset.data_omni_matrix.shape[1],
                            output_size_fism2_flare=20,
                            output_size_fism2_daily=20,
                            output_size_omni=20
                            )
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
    model.to(device)

    if torch.cuda.device_count()>1:
        print(f"Parallelizing the model on {torch.cuda.device_count()} GPUs")
        model=nn.DataParallel(model)

    benchmark = Benchmark(
        batch_size=opt.batch_size,
        num_workers=opt.num_workers,
        data_directory=opt.data_directory,
        output_directory=opt.model_folder,
        model_name=opt.model_name).evaluate_model(dataset, model)
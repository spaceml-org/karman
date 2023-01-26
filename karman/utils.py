import pandas as pd
import torch
import torch
from torch.utils.data import DataLoader
import numpy as np
import karman
from karman.nn import *
import os
from karman import Benchmark


def load_model(model_path,
               data_directory='/home/jupyter/karman-project/data_directory'):

    print('Loading Data')
    model_opt=torch.load(model_path)['opt']
    print(model_opt)

    dataset=karman.ThermosphericDensityDataset(
        directory=data_directory,
        features_to_exclude_thermo=model_opt.features_to_exclude_thermo.split(','),
        create_cyclical_features=True,
        max_altitude=model_opt.max_altitude
    )

    print('Loading Model')

    if model_opt.model == 'SimpleNN':
        model = SimpleNN().to(dtype=torch.float32)

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

    # infer model fold from path given
    # assumes model follows format: 'model_name_fold_x_seed_0' and uses 'x' as the fold
    fold = int(model_path.split('fold')[-1][1])

    test_month_idx = 2 * (int(fold) - 1)
    validation_month_idx = test_month_idx + 2
    dataset._set_indices(test_month_idx=[test_month_idx], validation_month_idx=[validation_month_idx])

    #Need to do this if using the LazyLinear Module (avoids having to hard code input layers to a linear layer)
    dummy_loader = torch.utils.data.DataLoader(dataset.validation_dataset(),
                                               batch_size=2,
                                               pin_memory=False,
                                               num_workers=0,
                                               drop_last=True)
    model.forward(next(iter(dummy_loader)))
    return model, dataset

def benchmark_model(model_path,
                    batch_size=512,
                    num_workers=30,
                    data_directory='/home/jupyter/karman-project/data_directory',
                    output_save='/home/jupyter/karman-project/benchmark_results',
                    model_name='model'):
    """
    Given a model path, this method loads the model and benchmarks for that stored model.
    """
    model, dataset = load_model(model_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    return Benchmark(
        batch_size=batch_size,
        num_workers=num_workers,
        data_directory=data_directory,
        output_directory=output_save,
        model_name=model_name).evaluate_model(dataset, model)

def benchmark_model_folder(folder_path,
                           batch_size=512,
                           num_workers=30,
                           data_directory='/home/jupyter/karman-project/data_directory',
                           output_save='/home/jupyter/karman-project/benchmark_results',
                           model_name='model'):
    """
    This method takes all models in a folder and benchmarks them. These models should all be
    the same model but trained on different folds.

    It assumes the models end in 'fold_x_seed_y'
    """
    model_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f[-6:-2] == 'seed']
    print('Models to load:', model_paths)
    results = []

    for model_path in model_paths:
        results.append(
            benchmark_model(model_path,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            data_directory=data_directory,
                            output_save=output_save,
                            model_name=model_name)
            )

    benchmark_results = pd.concat(results, ignore_index=True)
    benchmark_results_mean = benchmark_results.dropna().groupby(['Model', 'Metric Type', 'Condition']).mean()
    benchmark_results_std = benchmark_results.dropna().groupby(['Model', 'Metric Type', 'Condition']).std()
    return benchmark_results_mean, benchmark_results_std
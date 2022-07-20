from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import numpy as np
import karman
from karman import FeedForwardDensityPredictor
import os
from torch.utils.data import Subset

class Benchmark():
    def __init__(self,
                 batch_size=512,
                 num_workers=12,
                 data_directory='/home/jupyter/data'):
        self.density_column = 'tudelft_thermo__ground_truth_thermospheric_density__[kg/m**3]'
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_directory = data_directory

    def evaluate_model(self, dataset, model):
        """
        Evaluates the model on the dataset with multiple metrics
        """
        print("\nEvaluating Model.")

        predictions, targets = self.get_predictions_and_targets(
            dataset,
            model
        )

        #TODO Add model evaluation for storm conditions etc.
        #TODO Add JB08 and NRLMSISE results to the benchmarking results.
        print('Model RMSE:', rmse(predictions, targets))
        print('Model MPE:', mpe(predictions, targets))

        return {
            'Model RMSE': rmse(predictions, targets),
            'Model MPE': mpe(predictions, targets)
        }


    def get_predictions_and_targets(self, dataset, model):
        """
        Gather up the predictions and targets from the dataset.
        """
        targets = []
        predictions = []

        #TODO need to find away around this indices reading, its horrible
        with open(os.path.join(self.data_directory, "test_indices.txt"), 'r') as f:
            test_indices = [int(line.rstrip()) for line in f]
        loader = DataLoader(Subset(dataset, test_indices),
                            batch_size=self.batch_size,
                            num_workers=self.num_workers)

        with torch.no_grad():
            for batch in tqdm(loader):
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
        return predictions, targets


def mpe(target, prediction):
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


if __name__ == '__main__':
    dataset = karman.ThermosphericDensityDataset(
        directory='/home/jupyter/data',
        exclude_omni=True,
        exclude_fism2=True,
    )
    model_save_path = '/home/jupyter/karman-project/output_directory/best_model_FeedForwardDensityPredictor_2022-07-20 12:00:14.014884'
    model = FeedForwardDensityPredictor(len(dataset.data_thermo.columns))
    model.load_state_dict(torch.load(model_save_path))
    benchmark = Benchmark(batch_size=512, num_workers=50).evaluate_model(dataset, model)
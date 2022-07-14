import argparse
import os
import pprint
import sys
import time

import datetime
import karman
from karman import FFNN
import numpy as np
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm

def run():
    print('Karman training script')
    print(f'Version {karman.__version__}\n')

    parser = argparse.ArgumentParser(description='Karman', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', help='Random number seed', default=1, type=int)
    parser.add_argument('--model', help='Models to use', default='FFNN', choices=['FFNN'])
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--output_directory', help='Output directory', default='output_directory')
    parser.add_argument('--epochs', '-n', help='Number of epochs', default=10, type=int)
    parser.add_argument('--valid_every', default=1500, type=int)
#    parser.add_argument('--gpus', default=0, type=int)
    parser.add_argument('--learning_rate', help='learning rate to use', default=1e-4, type=float)
    parser.add_argument('--weight_decay', help='Weight decay: optimizer parameter', default=0., type=float)
    parser.add_argument('--optimizer', help='Optimizer to use', default='adam', choices=['adam', 'sgd'])
    parser.add_argument('--load_indices',help='If False, then train, validation and test set are computed on the fly; otherwise, they are loaded', default=True, type=bool)

    opt = parser.parse_args()

    print('Arguments:\n{}\n'.format(' '.join(sys.argv[1:])))
    print('Config:')
    pprint.pprint(vars(opt), depth=2, width=1)
    print()

    isExist = os.path.exists(opt.output_directory)

    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(opt.output_directory)
        print(f"Created directory for storing results: {opt.output_directory}")

    dataset=karman.ThermosphericDensityDataset(lag_minutes_omni=10, lag_days_fism2_daily=10, lag_minutes_fism2_flare=150)
    
    print(f"Train, Valid, Test split:")
    if opt.load_indices==False:
        years = list(range(2003, 2022))
        months = np.array(range(1,13))
        train_idx = [0,1,2,5,6,9,10,11]
        validation_idx = [3,7]
        test_idx = [4,8]
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
            with open("train_indices.txt", 'w') as output:
                for row in train_indices:
                    output.write(str(row) + '\n')
            with open("val_indices.txt", 'w') as output:
                for row in val_indices:
                    output.write(str(row) + '\n')
            with open("test_indices.txt", 'w') as output:
                for row in test_indices:
                    output.write(str(row) + '\n')
    else:
        with open('train_indices.txt') as f:
            train_indices = [int(line.rstrip()) for line in f]
        with open('val_indices.txt') as f:
            val_indices = [int(line.rstrip()) for line in f]
        with open('test_indices.txt') as f:
            test_indices = [int(line.rstrip()) for line in f]

    print(f"Train set proportion: {len(train_indices)/len(dataset)*100} %")
    print(f"Validation set proportion: {len(val_indices)/len(dataset)*100} %")
    print(f"Test set proportion: {len(test_indices)/len(dataset)*100} %")

    #I perform the dataset split, creating train, valid, test dataloaders:
    train_sampler=SubsetRandomSampler(train_indices)
    valid_sampler=SubsetRandomSampler(val_indices)
    test_sampler=SubsetRandomSampler(test_indices)

    train_loader = torch.utils.data.DataLoader(dataset,
                                               batch_size=opt.batch_size,
                                               sampler=train_sampler,
                                               pin_memory=True,
                                               num_workers=opt.num_workers)
    valid_loader = torch.utils.data.DataLoader(dataset,
                                               batch_size=opt.batch_size,
                                               sampler=valid_sampler,
                                               pin_memory=True,
                                               num_workers=opt.num_workers)
    test_loader = torch.utils.data.DataLoader(dataset,
                                               batch_size=opt.batch_size,
                                               sampler=test_sampler,
                                               pin_memory=True,
                                               num_workers=opt.num_workers)
    
    model = FFNN(num_features=len(dataset[0][0]))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device is: {device}")
    model.to(device)
    if opt.optimizer=='adam':
        optimizer=optim.Adam(model.parameters(), lr=opt.learning_rate, weight_decay=opt.weight_decay)
    elif opt.optimizer=='sgd':
        optimizer=optim.SGD(model.parameters(), lr=opt.learning_rate, weight_decay=opt.weight_decay)
    model.train()
    train_losses=[]
    valid_losses=[]
    avg_valid_losses=[]
    time_start=datetime.datetime.now()

    i_total=0
    last_model_path=os.path.join(opt.output_directory,"last_model_"+opt.model+f"_{time_start}")
    best_model_path=os.path.join(opt.output_directory,"best_model_"+opt.model+f"_{time_start}")
    for epoch in range(opt.epochs):
        for batch_index, (inp,target) in enumerate(train_loader):
            inp,target=inp.to(device),target.to(device)
            optimizer.zero_grad()
            output = model(inp)
            train_loss=nn.MSELoss()(output,target.unsqueeze(1),)
            train_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_losses.append(float(train_loss))
            print((epoch, float(train_loss)),end='\r')

            if i_total%opt.valid_every==0:
                print("Validation\n")
                model.train(False)
                batches_valid_loss=0
                for batch_index_val, (inp_val, target_val) in enumerate(valid_loader):
                    inp_val, target_val=inp_val.to(device), target_val.to(device)
                    optimizer.zero_grad()
                    output_val=model(inp_val)
                    valid_loss=nn.MSELoss()(output_val target_val.unsqueeze(1),)
                    batches_valid_loss+=float(valid_loss)
                    valid_losses.append(float(valid_loss))

                batches_valid_loss=batches_valid_loss/len(test_loader)
                avg_valid_losses.append(batches_valid_loss)
                if batches_valid_loss<=min(avg_valid_losses):
                    print(f"Saving best model to: {best_model_path} \n")
                    torch.save(model.state_dict(), best_model_path)
                model.train(True)
            i_total+=1
    print(f"Saving last model to: {last_model_path}\n")
    torch.save(model.state_dict(), last_model_path)
    
if __name__ == "__main__":
    time_start = time.time()
    run()
    print('\nTotal duration: {}'.format(time.time() - time_start))
    sys.exit(0)
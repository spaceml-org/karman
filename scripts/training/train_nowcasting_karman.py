import pandas as pd
from sklearn.preprocessing import QuantileTransformer
from torch.utils.data import RandomSampler, SequentialSampler
import torch
import numpy as np
import argparse
import os
from pyfiglet import Figlet
from termcolor import colored
from tqdm import tqdm
import wandb
import torch
import pprint
import time
import sys
sys.path.append('../')
import karman

def mean_absolute_percentage_error(y_pred,y_true):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def mse(y_pred,y_true):
    return np.mean((y_true - y_pred) ** 2)

def train():
    print('Karman Model Training')
    f = Figlet(font='5lineoblique')
    print(colored(f.renderText('KARMAN 2.0'), 'red'))
    f = Figlet(font='digital')
    print(colored(f.renderText("Training Nowcasting Model"), 'blue'))
    print(colored(f'Version {karman.__version__}\n','blue'))
    parser = argparse.ArgumentParser(description='HL-24 Karman Model Training', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--device', type=str, default='cpu', help='Device to use for training')
    parser.add_argument('--torch_type', type=str, default='float32', help='Torch type to use for training')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--normalization_dict_path', type=str, default=None, help='Path to the normalization dictionary. If None, the normalization values are computed on the fly')
    parser.add_argument('--model_path', type=str, default=None, help='Path to the model to load. If None, a new model is created')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for the optimizer')
    parser.add_argument('--run_name', default='', help='Run name to be stored in wandb')
    parser.add_argument('--thermo_path', default='../data/satellites_data_w_sw_2mln.csv', help='Path to the thermo dataset. Default is ../data/satellites_data_w_sw.csv')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs to train the model')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for the dataloader')
    parser.add_argument('--min_date', type=str, default='2000-07-29 00:59:47', help='Min date to consider for the dataset')
    parser.add_argument('--max_date', type=str, default='2024-05-31 23:59:32', help='Max date to consider for the dataset')
    parser.add_argument('--hidden_layer_dim', type=int, default=48, help='Hidden layer dimension')
    parser.add_argument('--hidden_layers', type=int, default=3, help='Number of hidden layers')
    parser.add_argument('--train_type', type=str, default='log_exp_residual', choices= ['log_density', 'log_exp_residual'], help='Training type, currently supports either log_density or log_exp_residual')
    #make wandb active default to True and if provided to False:
    parser.add_argument('--wandb_active', action='store_true', help='Activate WandB logging')

    opt = parser.parse_args()
    if opt.wandb_active:    
        wandb.init(project='karman', config=vars(opt))
        # wandb.init(mode="disabled")
        if opt.run_name != '':
            wandb.run.name = opt.run_name
            wandb.run.save()
    print('Arguments:\n{}\n'.format(' '.join(sys.argv[1:])))
    print('Config:')
    pprint.pprint(vars(opt), depth=2, width=1)
    print()


    if 'float32':
        torch_type=torch.float32
    elif 'float64':
        torch_type=torch.float64
    else:
        raise ValueError('Invalid torch type. Only float32 and float64 are supported')
    torch.set_default_dtype(torch_type)
    
    #features_to_exclude_thermo=[
    #                            "all__dates_datetime__",
    #                            "tudelft_thermo__satellite__",
    #                            "tudelft_thermo__ground_truth_thermospheric_density__[kg/m**3]",
    #                            "all__year__[y]",
    #                            "NRLMSISE00__thermospheric_density__[kg/m**3]"]
    #if opt.train_type == 'log_exp_residual':
    #    features_to_exclude_thermo+=["tudelft_thermo__altitude__[m]"]


    karman_dataset=karman.KarmanDataset(thermo_path=opt.thermo_path,
                                min_date=pd.to_datetime(opt.min_date),
                                max_date=pd.to_datetime(opt.max_date),
                                normalization_dict=opt.normalization_dict_path,
                                #omni_path='omniweb_1min_data_2001_2022.h5',
                            )

    input_dimension=karman_dataset[0]['instantaneous_features'].shape[0]
    if opt.device=='cuda':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device=torch.device('cpu')    
    print(f'Device is {device}')
    
    karman_model=karman.SimpleNetwork( input_dim=input_dimension,
                                        act=torch.nn.LeakyReLU(negative_slope=0.01),
                                        hidden_layer_dims=[opt.hidden_layer_dim]*opt.hidden_layers,
                                        output_dim=1).to(device)
    #if the model path is passed, load from there:
    if opt.model_path is not None:
        karman_model.load_state_dict(torch.load(opt.model_path))

    num_params=sum(p.numel() for p in karman_model.parameters() if p.requires_grad)
    print(f'Karman model num parameters: {num_params}')

    #Train, validation, test splits:
    idx_test_fold=2
    test_month_idx = 2 * (idx_test_fold - 1)
    validation_month_idx = test_month_idx + 2
    print(test_month_idx,validation_month_idx)
    karman_dataset._set_indices(test_month_idx=[test_month_idx], validation_month_idx=[validation_month_idx],custom={2001: {"validation":2,"test":3},
                                                                                                                     2003: {"validation":9, "test":10},
                                                                                                                     2005: {"validation":4, "test":5},
                                                                                                                     2012: {"validation":8, "test":9},
                                                                                                                     2013: {"validation":4, "test":5},
                                                                                                                     2015: {"validation":2, "test":3},
                                                                                                                     2022: {"validation":0, "test":1},
                                                                                                                     2024: {"validation":3,"test":4}})
    train_dataset = karman_dataset.train_dataset()
    validation_dataset = karman_dataset.validation_dataset()
    test_dataset = karman_dataset.test_dataset()
    print(f'Training dataset example: {train_dataset[0].items()}')

    train_sampler = RandomSampler(train_dataset, num_samples=len(train_dataset))
    validation_sampler = RandomSampler(validation_dataset, num_samples=len(validation_dataset))
    test_sampler = SequentialSampler(test_dataset)

    ####### Training Parameters ##########
    batch_size = opt.batch_size
    # Here we set the optimizer
    optimizer = torch.optim.Adam(karman_model.parameters(), lr=opt.lr,amsgrad=True)
    #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[25,50,75,100,125,150,175,200,225,230,240,250,260,270], gamma=0.8, verbose=False)
    criterion=torch.nn.MSELoss()

    # And the dataloader
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        pin_memory=False,
        num_workers=opt.num_workers,
        sampler=train_sampler,
        drop_last=True,
    )
    validation_loader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=batch_size,
        pin_memory=False,
        num_workers=opt.num_workers,
        sampler=validation_sampler,
        drop_last=False,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        pin_memory=False,
        num_workers=opt.num_workers,
        sampler=test_sampler,
        drop_last=False,
    )

    losses_per_minibatch={'nn_mse_train':[],'nrlmsise00_mse_train':[],'nn_mape_train':[],'nrlmsise00_mape_train':[],
                        'nn_mse_valid':[],'nrlmsise00_mse_valid':[],'nn_mape_valid':[],'nrlmsise00_mape_valid':[]}
    losses_total={'nn_mse_train':[],'nrlmsise00_mse_train':[],'nn_mape_train':[],'nrlmsise00_mape_train':[],
                'nn_mse_valid':[],'nrlmsise00_mse_valid':[],'nn_mape_valid':[],'nrlmsise00_mape_valid':[]}

    # Training loop

    best_loss_total = np.inf
    best_loss = np.inf
    for epoch in range(opt.epochs):
        #first training loop:
        loss_total_nn=0.
        loss_total_nrlmsise00=0.
        mape_total_nn=0.
        mape_total_nrlmsise00=0.
        #we set the model in training mode:
        karman_model.train()
        for batch_idx,el in enumerate(train_loader):
            minibatch=el['instantaneous_features'].to(device)
            #let's store the normalized and unnormalized target density:
            target=el['target'].to(device)
            rho_target=el['ground_truth'].detach().cpu().numpy()
            #now the normalized and unnormalized NN-predicted density:
            if opt.train_type=='log_exp_residual':
                out_nn=torch.tanh(karman_model(minibatch).squeeze())
                target_nn=karman_dataset.scale_density(el['exponential_atmosphere'].to(device))+out_nn
            else:
                target_nn=karman_model(minibatch).squeeze()
            rho_nn=karman_dataset.unscale_density(target_nn.detach().cpu()).numpy()
            #finally the NRLMSISE-00 ones:
            rho_nrlmsise00=el['nrlmsise00'].detach().cpu().numpy()
            target_nrlmsise00=karman_dataset.scale_density(el['nrlmsise00'].to(device))
            #now the loss computation:
            loss_nn = criterion(target_nn, target)
            loss_nrlmsise00 = mse(target_nrlmsise00.detach().cpu().numpy(), target.detach().cpu().numpy())

            # Zeroes the gradient 
            optimizer.zero_grad()

            # Backward pass: compute gradient of the loss with respect to model parameters
            loss_nn.backward()

            # Calling the step function on an Optimizer makes an update to its parameters
            optimizer.step()

            #We compute the logged quantities
            #log to wandb:
            if opt.wandb_active:    
                wandb.log({'nn_mse_train':loss_nn.item(),'nrlmsise00_mse_train':loss_nrlmsise00,
                            'nn_mape_train':mean_absolute_percentage_error(rho_nn, rho_target),
                            'nrlmsise00_mape_train':mean_absolute_percentage_error(rho_nrlmsise00, rho_target)})
                
            losses_per_minibatch['nn_mse_train'].append(loss_nn.item())
            losses_per_minibatch['nrlmsise00_mse_train'].append(loss_nrlmsise00)
            losses_per_minibatch['nn_mape_train'].append(mean_absolute_percentage_error(rho_nn, rho_target))
            losses_per_minibatch['nrlmsise00_mape_train'].append(mean_absolute_percentage_error(rho_nrlmsise00, rho_target))
            #now let's also accumulate them for the overall loss computation in each epoch:
            loss_total_nn+=losses_per_minibatch['nn_mse_train'][-1]
            loss_total_nrlmsise00+=losses_per_minibatch['nrlmsise00_mse_train'][-1]
            mape_total_nn+=losses_per_minibatch['nn_mape_train'][-1]
            mape_total_nrlmsise00+=losses_per_minibatch['nrlmsise00_mape_train'][-1]
            
            #Save the best model (this is wrong and should be done on the dataset):
            if loss_nn.item()<best_loss:    
                best_loss=loss_nn.item()

            #Print every 10 minibatches:
            #if batch_idx%10:    
            #    print(f'minibatch: {batch_idx}/{len(train_loader)}, best minibatch loss till now: {best_loss:.4e}, NN MSE: {losses_per_minibatch['nn_mse_train'][-1]:.10f}, nrlmsise00 MSE: {losses_per_minibatch['nrlmsise00_mse_train'][-1]:.10f}, NN MAPE: {losses_per_minibatch['nn_mape_train'][-1]:.3f}, nrlmsise00 MAPE: {losses_per_minibatch['nrlmsise00_mape_train'][-1]:.3f}', end='\r')
        #log to wandb:
        if opt.wandb_active:    
            wandb.log({'nn_mse_train_total':loss_total_nn/len(train_loader),'nrlmsise00_mse_train_total':loss_total_nrlmsise00/len(train_loader),
                            'nn_mape_train_total':mape_total_nn/len(train_loader),
                            'nrlmsise00_mape_train_total':mape_total_nrlmsise00/len(train_loader)})
        # over the whole dataset, we take the average of the minibatch losses:
        losses_total['nn_mse_train'].append(loss_total_nn/len(train_loader))
        losses_total['nrlmsise00_mse_train'].append(loss_total_nrlmsise00/len(train_loader))
        losses_total['nn_mape_train'].append(mape_total_nn/len(train_loader))
        losses_total['nrlmsise00_mape_train'].append(mape_total_nrlmsise00/len(train_loader))

        #Print at the end of the epoch
        #curr_lr = scheduler.optimizer.param_groups[0]['lr']
        print(" "*300, end="\r")    
        print("\nTraining")
        print(f"Epoch {epoch + 1}/{opt.epochs}, NN MSE (total): {losses_total['nn_mse_train'][-1]:.7f}, nrlmsise00 MSE (total): {losses_total['nrlmsise00_mse_train'][-1]:.7f}, NN MAPE (total): {losses_total['nn_mape_train'][-1]:.3f}, nrlmsise00 MAPE (total): {losses_total['nrlmsise00_mape_train'][-1]:.3f}")
        # Perform a step in LR scheduler to update LR
        #scheduler.step()
        
        #Validation loop:
        loss_total_nn=0.
        loss_total_nrlmsise00=0.
        mape_total_nn=0.
        mape_total_nrlmsise00=0.
        #let's switch the model to evaluation mode:
        karman_model.eval()
        with torch.no_grad():
            for batch_idx,el in enumerate(validation_loader):
                minibatch=el['instantaneous_features'].to(device)
                #let's store the normalized and unnormalized target density:
                target=el['target'].to(device)
                rho_target=el['ground_truth'].detach().cpu().numpy()
                #now the normalized and unnormalized NN-predicted density:
                #now the normalized and unnormalized NN-predicted density:
                if opt.train_type=='log_exp_residual':
                    target_nn=karman_dataset.scale_density(el['exponential_atmosphere'].to(device))+torch.tanh(karman_model(minibatch).squeeze())
                else:
                    target_nn=karman_model(minibatch).squeeze()
                rho_nn=karman_dataset.unscale_density(target_nn).detach().cpu().numpy()
                #finally the NRLMSISE-00 ones:
                rho_nrlmsise00=el['nrlmsise00'].detach().cpu().numpy()
                target_nrlmsise00=karman_dataset.scale_density(el['nrlmsise00'].to(device))
                #now the loss computation:
                loss_nn = criterion(target_nn, target)
                loss_nrlmsise00 = mse(target_nrlmsise00.detach().cpu().numpy(), target.detach().cpu().numpy())
                #log to wandb:
                if opt.wandb_active:    
                    wandb.log({'nn_mse_valid':loss_nn.item(),'nrlmsise00_mse_valid':loss_nrlmsise00,
                                'nn_mape_valid':mean_absolute_percentage_error(rho_nn, rho_target),
                                'nrlmsise00_mape_valid':mean_absolute_percentage_error(rho_nrlmsise00, rho_target)})
                #We compute the logged quantities
                losses_per_minibatch['nn_mse_valid'].append(loss_nn.item())
                losses_per_minibatch['nrlmsise00_mse_valid'].append(loss_nrlmsise00)
                losses_per_minibatch['nn_mape_valid'].append(mean_absolute_percentage_error(rho_nn, rho_target))
                losses_per_minibatch['nrlmsise00_mape_valid'].append(mean_absolute_percentage_error(rho_nrlmsise00, rho_target))
                #now let's also accumulate them for the overall loss computation in each epoch:
                loss_total_nn+=losses_per_minibatch['nn_mse_valid'][-1]
                loss_total_nrlmsise00+=losses_per_minibatch['nrlmsise00_mse_valid'][-1]
                mape_total_nn+=losses_per_minibatch['nn_mape_valid'][-1]
                mape_total_nrlmsise00+=losses_per_minibatch['nrlmsise00_mape_valid'][-1]
            #log to wandb:
            if opt.wandb_active:    
                wandb.log({'nn_mse_valid_total':loss_total_nn/len(validation_loader),'nrlmsise00_mse_valid_total':loss_total_nrlmsise00/len(validation_loader),
                                'nn_mape_valid_total':mape_total_nn/len(validation_loader),
                                'nrlmsise00_mape_valid_total':mape_total_nrlmsise00/len(validation_loader)})
                # over the whole dataset, we take the average of the minibatch losses:
            losses_total['nn_mse_valid'].append(loss_total_nn/len(validation_loader))
            losses_total['nrlmsise00_mse_valid'].append(loss_total_nrlmsise00/len(validation_loader))
            losses_total['nn_mape_valid'].append(mape_total_nn/len(validation_loader))
            losses_total['nrlmsise00_mape_valid'].append(mape_total_nrlmsise00/len(validation_loader))
        print("\nValidation")
        print(f"Epoch {epoch + 1}/{opt.epochs}, NN MSE (total): {losses_total['nn_mse_valid'][-1]:.7f}, nrlmsise00 MSE (total): {losses_total['nrlmsise00_mse_valid'][-1]:.7f}, NN MAPE (total): {losses_total['nn_mape_valid'][-1]:.3f}, nrlmsise00 MAPE (total): {losses_total['nrlmsise00_mape_valid'][-1]:.3f}")
        #updating torch best model:
        if losses_total['nn_mse_valid'][-1] < best_loss_total:
            #log to wandb:
            if opt.wandb_active:
                wandb.log({'best_nn_mse_valid':losses_total['nn_mse_valid'][-1]})
            #create directory if it does not exist:
            if not os.path.exists('../models'):
                os.makedirs('../models')
            torch.save(karman_model.state_dict(), f"../models/karman_model_{opt.train_type}_valid_loss_{losses_total['nn_mse_valid'][-1]}_params_{num_params}.torch")
            best_loss_total=losses_total['nn_mse_valid'][-1]

if __name__ == "__main__":
    time_start = time.time()
    train()
    print('\nTotal duration: {}'.format(time.time() - time_start))
    sys.exit(0)


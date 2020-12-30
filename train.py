import argparse
import logging
import os
import json
from pathlib import Path

import numpy as np
import pandas as pd 
import torch
from torch import nn
import torch.optim as optim

from AEncoder import AEncoder
from Preprocess import preprocessing, Time_Series_Split
from Net import MLP

def MODEL(params, X, y, AEncoder, splits, device=None):
  num_epochs = 10
  batch_size = 4096
  loss_fn = nn.BCELoss().to(device)
  config = {**params}

  Val_Loss = 0.0
  N_Samples = 0

  for fold, (train_idx, valid_idx) in enumerate(splits):
    print('Fold : {}'.format(fold))
    # Prepare datasets
    tr_x, tr_y = torch.tensor(X[train_idx], dtype=torch.float), torch.tensor(y[train_idx], dtype=torch.float)
    val_x, val_y = torch.tensor(X[valid_idx], dtype=torch.float), torch.tensor(y[valid_idx], dtype=torch.float)

    tr = torch.utils.data.TensorDataset(tr_x, tr_y)
    val = torch.utils.data.TensorDataset(val_x, val_y)
    train_loader = torch.utils.data.DataLoader(tr, batch_size=batch_size)
    val_loader = torch.utils.data.DataLoader(val, batch_size=batch_size)
    

    # define model
    model = MLP(config, AEncoder).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(num_epochs):      
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            #scheduler.step()

        epoch_loss = running_loss / len(train_loader.dataset)
  
        print('\t\t Training: Epoch({}) - Loss: {:.4f}'.format(epoch, epoch_loss))

        model.eval()   
        vrunning_loss = 0.0
        num_samples = 0
        for data, labels in val_loader:
            data = data.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            with torch.no_grad():
                outputs = model(data)
                loss = loss_fn(outputs, labels)
            vrunning_loss += loss.item() * data.size(0)
            num_samples += labels.size(0)
            
        vepoch_loss = vrunning_loss/num_samples
        print('\t\t Validation({}) - Loss: {:.4f}'.format(epoch, vepoch_loss))
    
    # update to global loss of the last epoch
    Val_Loss += loss.item() * data.size(0)
    # update global # of samples of the last epoch
    N_Samples += labels.size(0)

  return model


def _parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('output_dir', type=Path,
                        help='')
    args = parser.parse_args()
    return args



def main():
    args = _parse_args()
    with open(args.output_dir/ 'config.json') as f:
        params = json.load(f)

    model_dir = os.path.join('checkpoints','mlp.pkl')

    X, y, _ = preprocessing()
    train = pd.read_csv('data/train.csv')
    splits = Time_Series_Split(train, y)

    GPU = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #NUM_FEATURES = X.shape[1]

    AEncoder = AEncoder(X, y, train=False)

    mlp = MODEL(params, X, y, AEncoder, splits, device=GPU)

    torch.save(mlp.state_dict(), model_dir)
    
    return mlp

    



    
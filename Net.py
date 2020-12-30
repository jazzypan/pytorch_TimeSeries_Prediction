import numpy as np 
import pandas as pd 
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

SEED = 43
torch.manual_seed(SEED)
np.random.seed(SEED)

# MLP 
#------------------------------------------------------------------------------#
class MLP(nn.Module):
    def __init__(self, config, AEncoder):
        super(MLP, self).__init__()
        
        self.criterion = nn.BCELoss()
        self.AEncoder = AEncoder
        self.lr = config["lr"]

        drop_out = [config[key] for key, v in config.items() if 'dropout' in key]
        hidden_size = [config[key] for key,v in config.items() if 'layer' in key]
      
        input_shape = 260
        
        layers = [] 
        
        for i in range(len(hidden_size)): 
            out_shape = hidden_size[i]
            
            layers.append(nn.Dropout(drop_out[i]))
            layers.append(nn.Linear(input_shape, out_shape))
            layers.append(nn.BatchNorm1d(out_shape))
            layers.append(nn.SiLU())  # SiLU aka swish
            input_shape = out_shape
        
        layers.append(nn.Dropout(drop_out[-1]))
        layers.append(nn.Linear(input_shape, 5))
        layers.append(nn.Sigmoid())
        
        self.model = torch.nn.Sequential(*layers)
    
    def encoder_decoder(self, x):
        encoded = self.AEncoder(x)
        decoded = self.AEncoder.decoder(encoded)
        return decoded

    def forward(self, x):
        
        # In lightning, forward defines the prediction/inference actions
        decoded = self.encoder_decoder(x)
        x = torch.cat((x, decoded), dim=1)
        x = self.model(x)
        return x


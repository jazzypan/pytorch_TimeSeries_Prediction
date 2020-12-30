import argparse
import logging
import os

import numpy as np
import torch
from torch import nn
import torch.optim as optim
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import pytorch_lightning as pl
import torch.nn.functional as F

from Preprocess import preprocessing

# AEncoder
#------------------------------------------------------------------------------#
class JaneStreetDataset:
    def __init__(self, dataset, targets):
        self.dataset = dataset
        self.targets = targets

    def __len__(self):
        return self.dataset.shape[0]

    def __getitem__(self, item):
        return {
            'x': torch.tensor(self.dataset[item, :], dtype=torch.float),
            'y': torch.tensor(self.targets[item], dtype=torch.float)
        }
#-------------------------------------------#    
class DataModule(pl.LightningDataModule):
    def __init__(self, data, targets, BATCH_SIZE, fold = None):
        super().__init__()
        self.BATCH_SIZE = BATCH_SIZE
        self.data = data
        self.targets = targets
        self.fold = fold
        
    def preapre_data(self):
        pass
    
    def setup(self, stage=None):
        
        train_data, train_targets = self.data, self.targets
        
        self.train_dataset = JaneStreetDataset(
            dataset = train_data, #train_data.values
            targets = train_targets
        )
        
    def train_dataloader(self):
        train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.BATCH_SIZE
        )
        return train_loader
    
    def valid_dataloader(self):
        
        return None
    
    def test_dataloader(self):
        return None

#-------------------------------------------#
# Encoder - Decoder
class LitAutoEncoder(pl.LightningModule):
    def __init__(self, input_shape):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.BatchNorm1d(input_shape),
            nn.Linear(input_shape, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        self.decoder = nn.Sequential(
            nn.Dropout(.2),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_shape)
        )

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        embedding = self.encoder(x)
        return embedding

    def training_step(self, batch, batch_idx):
        x = batch['x']
        #x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        return loss
    
    #def training_epoch_end(self, outputs):
        # TODO: add roc_auc
            
        #avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        #logs = {'train_loss': avg_loss}
        #return {'log': logs, 'progress_bar': logs}
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    

def AEncoder(X, y, train=False, EPOCHS = 10, BATCH_SIZE = 4096):
    model_dir = os.path.join('checkpoints','encoder.pkl')

    #X, y, _ = preprocessing()
    NUM_FEATURES = X.shape[1]


    early_stop_callback = EarlyStopping(
        monitor='train_loss', min_delta=0.00, patience=10, verbose=True, mode='min'
    )

    GPU = int(torch.cuda.is_available())

    if train:
        DataLoader = DataModule(data=X, targets=y, BATCH_SIZE=BATCH_SIZE)
        trainer = pl.Trainer(gpus=GPU, max_epochs=EPOCHS, weights_summary='full', callbacks=[early_stop_callback])
        AEncoder = LitAutoEncoder(input_shape=NUM_FEATURES)
        trainer.fit(AEncoder, DataLoader)
        torch.save(AEncoder.state_dict(), model_dir)
    else:
        AEncoder = LitAutoEncoder(input_shape=NUM_FEATURES)
        AEncoder.load_state_dict(torch.load(model_dir))
    
    return AEncoder



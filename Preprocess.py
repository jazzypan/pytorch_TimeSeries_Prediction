import numpy as np 
import pandas as pd 
from tqdm import tqdm
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from utils import PurgedGroupTimeSeriesSplit 
#from DataLoader import *


def preprocessing():
    # load data
    train = pd.read_csv('data/train.csv')

    # drop `date > 85` 
    train = train.query('date > 85').reset_index(drop = True) 

    #limit memory use
    train = train.astype({c: np.float32 for c in train.select_dtypes(include='float64').columns}) 

    # fill NANs
    train.fillna(train.mean(),inplace=True)

    # Preprocessing
    train = train.query('weight > 0').reset_index(drop = True)
    train['action'] = (train['resp'] > 0).astype('int')

    # features
    features = [c for c in train.columns if 'feature' in c]
    X = train[features].values

    # targets: multi-targets
    resp_cols = ['resp_1','resp_2','resp_3', 'resp_4','resp']
    y = np.stack([(train[c] > 0).astype('int') for c in resp_cols]).T #Multitarget

    # save feature means for submission
    f_mean = np.mean(train[features[1:]].values,axis=0)

    return X, y, f_mean


def Time_Series_Split(train, y, FOLDS=5):
    gkf = PurgedGroupTimeSeriesSplit(n_splits = FOLDS, group_gap=31)
    splits = list(gkf.split(y, groups=train['date'].values))
    return splits



    




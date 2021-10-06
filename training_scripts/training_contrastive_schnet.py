import torch
import torch.nn as nn
import torch_geometric
import numpy as np
import datetime
import scipy
import gzip
import math
import rdkit
import rdkit.Chem
from rdkit.Chem import TorsionFingerprints
from collections import defaultdict
import pandas as pd
from tqdm import tqdm
from copy import deepcopy
import random

import os
import sys
import json
from model.params_interpreter import string_to_object 

from model.gnn_3D.schnet import SchNet
from model.gnn_3D.dimenet_pp import DimeNetPlusPlus
from model.gnn_3D.spherenet import SphereNet

from torch_geometric.nn.acts import swish

from model.gnn_3D.train_functions import classification_loop, contrastive_loop, binary_ranking_regression_loop

from model.gnn_3D.train_models import train_classification_model, train_contrastive_model, train_binary_ranking_regression_model

from model.datasets_samplers import Dataset_3D_GNN, StereoBatchSampler, SiameseBatchSampler, Sample_Map_To_Positives, Sample_Map_To_Negatives, NegativeBatchSampler, SingleConformerBatchSampler

import sklearn

args = sys.argv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print('reading data...')

# READ HYPERPARAMETERS
with open(str(args[1])) as f: # args[1] should contain path to params.json file
    params = json.load(f)

seed = params['random_seed']
random.seed(seed)
np.random.seed(seed = seed)
torch.manual_seed(seed)
    
train_dataframe = pd.read_pickle(params['train_datafile'])
val_dataframe = pd.read_pickle(params['validation_datafile'])

if params['select_N_enantiomers']: # number of enantiomers to include for training; default = null 
    smiles_nostereo = list(set(train_dataframe.SMILES_nostereo))
    random.shuffle(smiles_nostereo)
    select_smiles_nostereo = smiles_nostereo[0:params['select_N_enantiomers']]
    train_dataframe = train_dataframe[train_dataframe.SMILES_nostereo.isin(select_smiles_nostereo)].sort_values('SMILES_nostereo').reset_index(drop = True)


# CREATE DIRECTORY FOR SAVING/CHECKPOINTING
save = params['save']

PATH = args[2] # should contain path to subfolder where files will be saved
if PATH[-1] != '/':
    PATH = PATH + '/'

if not os.path.exists(PATH) and save == True:
    os.makedirs(PATH)

#CREATE MODEL
random.seed(seed)
np.random.seed(seed = seed)
torch.manual_seed(seed)

print('creating model...')

model = SchNet(hidden_channels = params['hidden_channels'], # 128
               num_filters = params['num_filters'], # 128
               num_interactions = params['num_interactions'], # 6
               num_gaussians = params['num_gaussians'], # 50
               cutoff = params['cutoff'], # 10.0
               max_num_neighbors = params['max_num_neighbors'], # 32
               out_channels = params['out_channels'], # 1
               readout = 'add',
               dipole = False,
               mean = None,
               std = None,
               atomref = None, 
               MLP_hidden_sizes = [], # [] for contrastive
    )

if params['pretrained'] != "":
    print('loading pretrained weights...')
    model.load_state_dict(torch.load(params['pretrained'], map_location=next(model.parameters()).device), strict=False)

model.to(device)

# DEFINE OPTIMIZERS
lr = params['lr']
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr = lr)

# Choosing Loss
loss_function = params['loss_function']

# only for contrastive learning
margin = params['margin']

# only for docking
absolute_penalty = params['absolute_penalty'] # default is 1.0
relative_penalty = params['relative_penalty'] # default is None (null). If a float >=0.0, we have to use a SiameseBatchSampler

# BUILDING DATA LOADERS
batch_size = params['batch_size']

# only for SiameseBatchSampler
N_pos = params['N_pos']
N_neg = params['N_neg']
stratified = params['stratified']
withoutReplacement = params['withoutReplacement']

# only for StereoBatchSampler
grouping = params['grouping'] # one of ['none', 'stereoisomers', 'graphs']

# selecting iteration style
if params['iteration_mode'] == 'stereoisomers':
    single_conformer_train_dataframe = train_dataframe.groupby('ID').sample(1)
    single_conformer_val_dataframe = val_dataframe.groupby('ID').sample(1)
    
    BatchSampler_train = SingleConformerBatchSampler(single_conformer_train_dataframe,
                                              train_dataframe, 
                                              batch_size,
                                              N_pos = N_pos,
                                              N_neg = N_neg, 
                                              withoutReplacement = withoutReplacement, 
                                              stratified = stratified)

    BatchSampler_val = SingleConformerBatchSampler(single_conformer_val_dataframe,
                                              val_dataframe, 
                                              batch_size,
                                              N_pos = N_pos,
                                              N_neg = N_neg, 
                                              withoutReplacement = withoutReplacement, 
                                              stratified = stratified)
    
elif params['iteration_mode'] == 'conformers':
    BatchSampler_train = SiameseBatchSampler(train_dataframe, 
                                            batch_size,
                                            N_pos = N_pos,
                                            N_neg = N_neg, 
                                            withoutReplacement = withoutReplacement, 
                                            stratified = stratified)

    BatchSampler_val = SiameseBatchSampler(val_dataframe, 
                                            batch_size,
                                            N_pos = N_pos,
                                            N_neg = N_neg, 
                                            withoutReplacement = withoutReplacement, 
                                            stratified = stratified)
    

train_dataset = Dataset_3D_GNN(train_dataframe, 
                                    regression = '', # top_score, RS_label_binary, sign_rotation
                              )

val_dataset = Dataset_3D_GNN(val_dataframe, 
                                    regression = '', # top_score, RS_label_binary, sign_rotation
                            )

num_workers = params['num_workers']
train_loader = torch_geometric.data.DataLoader(train_dataset, batch_sampler = BatchSampler_train, num_workers = num_workers)
val_loader = torch_geometric.data.DataLoader(val_dataset, batch_sampler = BatchSampler_val, num_workers = num_workers)


# BEGIN TRAINING
weighted_sum = params['weighted_sum'] # only for StereoBatchSampler

if not os.path.exists(PATH + 'checkpoint_models') and save == True:
    os.makedirs(PATH + 'checkpoint_models')

N_epochs = params['N_epochs']

train_contrastive_model(model, 
                        train_loader, 
                        val_loader, 
                        N_epochs = N_epochs, 
                        optimizer = optimizer, 
                        device = device, 
                        loss_function = loss_function, 
                        batch_size = batch_size, 
                        margin = margin, 
                        save = save, 
                        PATH = PATH)

print('completed process')

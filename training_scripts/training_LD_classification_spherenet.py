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

from model.gnn_3D.train_functions import evaluate_binary_ranking_regression_loop, evaluate_classification_loop


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

CV_fold = params['CV_fold'] # 1, 2, 3, 4, or 5
dataframe = pd.read_pickle(params['train_datafile'])

# groups that contain mol objects that SphereNet fails to process
remove_groups = ['C=CC(COC(c1ccccc1)(c1ccccc1)c1ccccc1)NS(=O)(=O)c1ccccc1[N+](=O)[O-]',
 'COc1ccc(C2(N)C(=O)N(C(c3ccccc3)(c3ccccc3)c3ccccc3)c3cc(OC)ccc32)cc1',
 'COc1cccc(C(c2ccc[nH]2)(c2ccccc2OC)c2[nH]c3ccccc3c2-c2ccccc2)c1',
 'COc1ccccc1C(c1ccccc1)(c1ccc[nH]1)c1[nH]c2ccccc2c1-c1ccccc1',
 'Cc1ccc(C2(N)C(=O)N(C(c3ccccc3)(c3ccccc3)c3ccccc3)c3ccccc32)cc1',
 'O=C1OC2(C(=O)N(C(c3ccccc3)(c3ccccc3)c3ccccc3)c3ccccc32)n2c1cc1ccccc12']

dataframe = dataframe[~dataframe.SMILES_nostereo.isin(remove_groups)].reset_index(drop = True)

train_val_dataframe = dataframe[(dataframe.test_fold != CV_fold)] # test_dataframe = dataframe[(dataframe.test_fold == CV_fold)]

train_dataframe = train_val_dataframe[~train_val_dataframe['isVal_fold_'+str(CV_fold)]].sort_values(['SMILES_nostereo', 'ID']).reset_index(drop = True)
val_dataframe = train_val_dataframe[train_val_dataframe['isVal_fold_'+str(CV_fold)]].sort_values(['SMILES_nostereo', 'ID']).reset_index(drop = True)

if params['sample_1conformer'] == True:
    train_dataframe = train_dataframe.groupby('ID').sample(1, random_state = seed).sort_values('SMILES_nostereo').reset_index(drop = True)
    #val_dataframe = val_dataframe.groupby('ID').sample(1, random_state = seed).sort_values('SMILES_nostereo').reset_index(drop = True)

if params['select_N_enantiomers']: # number of enantiomers to include for training; default = null (i.e., for creating learning curves) 
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

model = SphereNet(
            energy_and_force = False, # False
            cutoff = params['cutoff'], # 5.0
            num_layers = params['num_layers'], # 4
            hidden_channels = params['hidden_channels'], # 128
            out_channels = params['out_channels'], # 1
            int_emb_size = params['int_emb_size'], # 64
            basis_emb_size_dist = params['basis_emb_size_dist'], # 8
            basis_emb_size_angle = params['basis_emb_size_angle'], # 8
            basis_emb_size_torsion = params['basis_emb_size_torsion'], # 8
            out_emb_channels = params['out_emb_channels'], # 256
            num_spherical = params['num_spherical'], # 7
            num_radial = params['num_radial'], # 6
            envelope_exponent = params['envelope_exponent'], # 5
            num_before_skip = params['num_before_skip'], # 1
            num_after_skip = params['num_after_skip'], # 2
            num_output_layers = params['num_output_layers'], # 3
            act=swish, 
            output_init='GlorotOrthogonal', 
            use_node_features = True,
            MLP_hidden_sizes = params['MLP_hidden_sizes'], # [] for contrastive
    )

if params['pretrained'] != "":
    print('loading pretrained weights...')
    model.load_state_dict(torch.load(params['pretrained'], map_location=next(model.parameters()).device), strict=False)

model.to(device)

# DEFINE OPTIMIZERS AND SCHEDULERS
lr = params['lr']
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr = lr)

# Choosing Loss
loss_function = params['loss_function']

# only for contrastive learning
margin = params['margin']

# only for docking (or ranking)
absolute_penalty = params['absolute_penalty'] # default is 1.0 (0.0 for ranking)
relative_penalty = params['relative_penalty'] # default is 0.0 or None (null); (1.0 for ranking; requires NegativeBatchSampler with only 1 negative per anchor). If a float > 0.0, we have to use a SiameseBatchSampler

# BUILDING DATA LOADERS
batch_size = params['batch_size']

# only for SiameseBatchSampler and its analogues
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
    BatchSampler_train = NegativeBatchSampler(train_dataframe, 
                                          batch_size, 
                                          N_neg = N_neg, 
                                          withoutReplacement = withoutReplacement, 
                                          stratified = stratified)

    BatchSampler_val = NegativeBatchSampler(val_dataframe, 
                                        batch_size,  
                                        N_neg = N_neg, 
                                        withoutReplacement = withoutReplacement, 
                                        stratified = stratified)


train_dataset = Dataset_3D_GNN(train_dataframe, 
                                    regression = 'sign_rotation', # top_score, RS_label_binary, sign_rotation
                              )

val_dataset = Dataset_3D_GNN(val_dataframe, 
                                    regression = 'sign_rotation', # top_score, RS_label_binary, sign_rotation
                            )

num_workers = params['num_workers']
train_loader = torch_geometric.data.DataLoader(train_dataset, batch_sampler = BatchSampler_train, num_workers = num_workers)
val_loader = torch_geometric.data.DataLoader(val_dataset, batch_sampler = BatchSampler_val, num_workers = num_workers)

# BEGIN TRAINING
weighted_sum = params['weighted_sum'] #  use for StereoBatchSampler or when batch sizes are not equivalent (e.g., for stratified triplet/negative sampling)

if not os.path.exists(PATH + 'checkpoint_models') and save == True:
    os.makedirs(PATH + 'checkpoint_models')

N_epochs = params['N_epochs']

best_state_dict = train_classification_model(model, 
                           train_loader, 
                           val_loader,
                           N_epochs = N_epochs, 
                           optimizer = optimizer, 
                           device = device, 
                           batch_size = batch_size, 
                           weighted_sum = weighted_sum, 
                           save = save,
                           PATH = PATH)

print('completed training')

print('evaluating model')

model.load_state_dict(best_state_dict)
model.to(device)

# cross validation
test_dataframe = dataframe[(dataframe.test_fold == CV_fold)].reset_index(drop = True)

test_dataset = Dataset_3D_GNN(test_dataframe, 
                                    regression = 'sign_rotation', # top_score, RS_label_binary, sign_rotation
                              )

test_loader = torch_geometric.data.DataLoader(test_dataset, num_workers = num_workers, batch_size = 1000, shuffle = False)

targets, outputs = evaluate_classification_loop(model, test_loader, device, batch_size = 1000, dataset_size = len(test_dataset))

results_df = deepcopy(test_dataframe[['ID', 'SMILES_nostereo', 'sign_rotation']])
results_df['targets'] = targets
results_df['outputs'] = outputs

results_df.to_csv(PATH + 'best_model_test_results.csv')

print('completed processes')

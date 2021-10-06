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

from model.alpha_encoder import Encoder

from model.train_functions import classification_loop_alpha, contrastive_loop_alpha, binary_ranking_regression_loop_alpha

from model.train_functions import evaluate_binary_ranking_regression_loop_alpha, evaluate_classification_loop_alpha

from model.train_models import train_classification_model, train_contrastive_model, train_binary_ranking_regression_model

from model.datasets_samplers import MaskedGraphDataset, StereoBatchSampler, SiameseBatchSampler, Sample_Map_To_Positives, Sample_Map_To_Negatives, NegativeBatchSampler, SingleConformerBatchSampler

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

if params['sample_1conformer'] == True:
    train_dataframe = train_dataframe.groupby('ID').sample(1, random_state = seed).sort_values('SMILES_nostereo').reset_index(drop = True)
    val_dataframe = val_dataframe.groupby('ID').sample(1, random_state = seed).sort_values('SMILES_nostereo').reset_index(drop = True)

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
layers_dict = deepcopy(params['layers_dict'])

activation_dict = deepcopy(params['activation_dict'])
for key, value in params['activation_dict'].items(): 
    activation_dict[key] = string_to_object[value] # convert strings to actual python objects/functions using pre-defined mapping

num_node_features = 52
num_edge_features = 14

model = Encoder(
    F_z_list = params['F_z_list'], # dimension of latent space
    F_H = params['F_H'], # dimension of final node embeddings, after EConv and GAT layers
    F_H_embed = num_node_features, # dimension of initial node feature vector, currently 41
    F_E_embed = num_edge_features, # dimension of initial edge feature vector, currently 12
    F_H_EConv = params['F_H_EConv'], # dimension of node embedding after EConv layer
    layers_dict = layers_dict,
    activation_dict = activation_dict,
    GAT_N_heads = params['GAT_N_heads'],
    chiral_message_passing = params['chiral_message_passing'],
    CMP_EConv_MLP_hidden_sizes = params['CMP_EConv_MLP_hidden_sizes'],
    CMP_GAT_N_layers = params['CMP_GAT_N_layers'],
    CMP_GAT_N_heads = params['CMP_GAT_N_heads'],
    c_coefficient_normalization = params['c_coefficient_normalization'], # None, or one of ['softmax']
    encoder_reduction = params['encoder_reduction'], #mean or sum
    output_concatenation_mode = params['output_concatenation_mode'], # none (if contrastive), conformer, molecule, or z_alpha (if regression)
    EConv_bias = params['EConv_bias'], 
    GAT_bias = params['GAT_bias'], 
    encoder_biases = params['encoder_biases'], 
    dropout = params['dropout'], # applied to hidden layers (not input/output layer) of Encoder MLPs, hidden layers (not input/output layer) of EConv MLP, and all GAT layers (using their dropout parameter)
    )

if params['pretrained'] != "":
    print('loading pretrained weights...')
    model.load_state_dict(torch.load(params['pretrained'], map_location=next(model.parameters()).device), strict=False)

model.to(device)

#SET UNLEARNABLE PARAMETERS
if params['c_coefficient_mode'] == 'random':
    for p in model.InternalCoordinateEncoder.Encoder_c.parameters():
        p.requires_grad = False
        
try:
    if params['phase_shift_coefficient_mode'] == 'random': # random or learned (if unspecified, will default to learned)
        for p in model.InternalCoordinateEncoder.Encoder_sinusoidal_shift.parameters():
            p.requires_grad = False
        print('not learning phase shifts...')
    elif params['phase_shift_coefficient_mode'] == 'learned':
        print('learning phase shifts...')
except:
    print('learning phase shifts...')
    pass

# DEFINE OPTIMIZERS
lr = params['default_lr']
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr = lr)
optimizers = [optimizer]

# Choosing Loss
loss_function = params['loss_function']

# only for contrastive learning
margin = params['margin']
contrastive_vector = params['contrastive_vector']

# only for docking
absolute_penalty = params['absolute_penalty'] # default is 1.0
relative_penalty = params['relative_penalty'] # default is None (null). If a float >=0.0, we have to use a SiameseBatchSampler

# BUILDING DATA LOADERS
batch_size = params['batch_size']
stereoMask = params['stereoMask']

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

train_dataset = MaskedGraphDataset(train_dataframe, 
                                    regression = 'RS_label_binary', # top_score, RS_label_binary, sign_rotation
                                    stereoMask = stereoMask,
                                    mask_coordinates = params['mask_coordinates'], 
                                    )

val_dataset = MaskedGraphDataset(val_dataframe, 
                                    regression = 'RS_label_binary', # top_score, RS_label_binary, sign_rotation
                                    stereoMask = stereoMask,
                                    mask_coordinates = params['mask_coordinates'], 
                                    )

num_workers = params['num_workers']
train_loader = torch_geometric.data.DataLoader(train_dataset, batch_sampler = BatchSampler_train, num_workers = num_workers)
val_loader = torch_geometric.data.DataLoader(val_dataset, batch_sampler = BatchSampler_val, num_workers = num_workers)


# BEGIN TRAINING
weighted_sum = params['weighted_sum'] # only for StereoBatchSampler

if not os.path.exists(PATH + 'checkpoint_models') and save == True:
    os.makedirs(PATH + 'checkpoint_models')

N_epochs = params['N_epochs']
auxillary_torsion_loss = params['auxillary_torsion_loss']

best_state_dict = train_classification_model(model, 
                           train_loader, 
                           val_loader,
                           N_epochs = N_epochs, 
                           optimizers = optimizers, 
                           device = device, 
                           batch_size = batch_size, 
                           auxillary_torsion_loss = auxillary_torsion_loss,
                           weighted_sum = weighted_sum, 
                           save = save,
                           PATH = PATH)

print('completed training')

print('evaluating model')

model.load_state_dict(best_state_dict)
model.to(device)

# cross validation
test_dataframe = pd.read_pickle(params['test_datafile'])

test_dataset = MaskedGraphDataset(test_dataframe, 
                                    regression = 'RS_label_binary', # top_score, RS_label_binary, sign_rotation
                                    stereoMask = params['stereoMask'],
                                    mask_coordinates = params['mask_coordinates'], 
                                    )

test_loader = torch_geometric.data.DataLoader(test_dataset, num_workers = num_workers, batch_size = 1000, shuffle = False)

targets, outputs = evaluate_classification_loop_alpha(model, test_loader, device, batch_size = 1000, dataset_size = len(test_dataset))

results_df = deepcopy(test_dataframe[['ID', 'SMILES_nostereo', 'RS_label_binary']])
results_df['targets'] = targets
results_df['outputs'] = outputs

results_df.to_csv(PATH + 'best_model_test_results.csv')

print('completed processes')
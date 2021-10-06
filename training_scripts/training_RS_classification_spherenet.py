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
    
train_dataframe = pd.read_pickle(params['train_datafile'])
val_dataframe = pd.read_pickle(params['validation_datafile'])

# groups that contain mol objects that SphereNet fails to process
remove_groups = ['CCc1cc2c(cc1C(F)(F)F)N(CCNC(=O)OC(C)(C)C)CCCC2N(Cc1cc(C(F)(F)F)cc(C(F)(F)F)c1)c1nnn(C)n1', 'COc1ccccc1C1C(C(=O)OC(C)C)=C(C(F)(F)F)N=C2SC=C(CC(=O)NCc3ccncc3)N21', 'COc1ccc(S(=O)(=O)N(Cc2c(C(C)C)nn(-c3ccccc3)c2Oc2cccc(F)c2)CC2CCCO2)cc1', '[2H]c1c([2H])c([2H])c(C(Oc2c([2H])c([2H])c([2H])c([2H])c2C([2H])([2H])[2H])C([2H])([2H])C([2H])([2H])NC([2H])([2H])[2H])c([2H])c1[2H]', 'CCCC(OC(=O)C(Cc1ccc(O)c(C(C)(C)C)c1)C(=O)OC(C)(C)C)(C1CC(C)(C)N(C)C(C)(C)C1)C1CC(C)(C)N(C)C(C)(C)C1', 'CCOc1c(OC)cccc1C1C(C(=O)OCCOC)=C(C)N=c2sc(=Cc3c(OCc4ccccc4)ccc4ccccc34)c(=O)n21', 'CCN1C(=O)C2=C(CC13CC3)OC1=C(C(=O)CC(C)(C)C1)C2c1c(OC)ccc(-c2ccc(C)c(C(=O)O)n2)c1C', 'CC1=CC(C)(c2c(C)c(N=[N+]=[N-])c3c(c2-c2cccc4c2Cc2ccccc2-4)-c2c(cccc2C2CC2)C3)C(C)=C1C', 'COc1ccc(C(c2[nH]ncc2C=C(Cc2cc3c(cc2OC)OCO3)C(=O)O)C2CC2)c(OCc2ccccc2C(=O)O)c1', 'CCNC(=O)C(C)N(Cc1ccccc1F)C(=O)CN(c1ccc(Cl)c(C(F)(F)F)c1)S(=O)(=O)c1ccc(C)cc1', 'O=S(=O)(OC(OS(=O)(=O)c1ccc2ccccc2c1)(OS(=O)(=O)c1cccc2ccccc12)c1ccccc1)c1ccccc1', 'CCC(C(=O)NC(C)C)N(Cc1ccc(OC)cc1)C(=O)CN(c1ccccc1OC)S(=O)(=O)c1ccccc1', 'FC(F)(F)N(C(F)(F)C(F)(F)F)C(F)(F)C(F)(C(F)(F)F)C(F)(F)C(F)(C(F)(F)F)C(F)(F)F', 'COc1cc(CC(=O)NN(c2ccccc2N2CCCCC2)c2ccccc2S(=O)c2ccccc2)c(Br)cc1C(=O)O', 'COC(=O)c1ccccc1NC(=O)C1CN(S(=O)(=O)c2ccccc2)CCN1S(=O)(=O)c1ccccc1', '[2H]c1c([2H])c([2H])c(C([2H])(CC([2H])([2H])NC([2H])([2H])[2H])Oc2c([2H])c([2H])c([2H])c([2H])c2C([2H])([2H])[2H])c([2H])c1[2H]', 'Cc1cc2c(cc1C(F)(F)F)N(Cc1ccc(CC(=O)O)cc1)CCCC2N(Cc1cc(C(F)(F)F)cc(C(F)(F)F)c1)c1nnn(C)n1', 'CCOC(=O)C1=C(c2ccccc2)N=C2SC=C(CC(=O)NCCc3ccccn3)N2C1c1ccccc1OC', 'CC(C(=O)NC(C)(C)C)N(Cc1ccc(Cl)c(Cl)c1)C(=O)CN(c1cc(C(F)(F)F)ccc1Cl)S(=O)(=O)c1ccccc1', 'c1ccc(-n2c3cc4c(cc3c3ccc5ccccc5c32)c2ccccc2n4C2=Nc3ccccc3NC2c2ccc3ccccc3c2)cc1', 'CC(C(=O)NCc1ccc(Cl)cc1Cl)N(Cc1ccc(F)cc1)C(=O)CN(Cc1ccccc1)S(C)(=O)=O', 'CCOC(=O)C1=C(C)NC(SCC(=O)Nc2cc(C)ccc2C)=NC1c1cc(OC)c(OC)c(OC)c1', 'COc1ccc(CN(C(=O)CCc2ccc(S(=O)(=O)NC3CC3)cc2)C(C)C(=O)NCc2ccc3c(c2)OCO3)cc1', 'FC(F)(F)C(F)(F)OC(F)(C(F)(F)F)C(F)(C(F)(C(F)(F)F)C(F)(F)F)C(F)(C(F)(F)F)C(F)(F)F', 'COc1cc(C2N=C(SCc3cccc4ccccc34)NC(C)=C2C(=O)Nc2ccccc2)cc(OC)c1OC', 'CCC(C(=O)NC(C)C)N(Cc1cccc(OC)c1)C(=O)CN(c1ccc(F)c(Cl)c1)S(=O)(=O)c1ccccc1', 'CCC(C(=O)NC(C)C)N(Cc1ccc(OC)cc1)C(=O)CN(c1cc(Cl)ccc1Cl)S(=O)(=O)c1ccc(C)cc1', 'CCNC(=O)C(C)N(Cc1ccccc1F)C(=O)CN(c1ccc(Cl)c(C(F)(F)F)c1)S(=O)(=O)c1ccccc1', 'CNC(=O)C(C)N(Cc1ccc(OC)cc1)C(=O)CN(c1cccc(Cl)c1Cl)S(=O)(=O)c1ccc(C)cc1', 'COc1cc(C2C3=C(CC(C)(C)CC3=O)Nc3ccccc3N2CC(=O)N2CCC(C(N)=O)CC2)cc(OC)c1OC', 'CCc1nn(-c2ccc(F)cc2)c(Oc2cccc(C)c2)c1CN(CC1CCCO1)S(=O)(=O)c1ccc(OC)cc1', 'CN(C)CCN(Cc1ccccc1)C(=O)c1ccccc1CN1CC(CC(=O)O)CN(Cc2ccc(Cl)cc2Cl)C1=O', 'COc1ccc(CN(C(=O)c2ccc(Br)c(OC)n2)C(CON2C(=O)c3ccccc3C2=O)c2cc3c(C)cccc3o2)cc1', 'Cc1cc(C(=O)CSc2nnc(-c3cccnc3)n2-c2cccc(C(F)(F)F)c2)c(C)n1CC1CCCO1', 'CCNC(=O)C(C)N(Cc1cccc(Cl)c1)C(=O)CN(c1cccc(C(F)(F)F)c1)S(=O)(=O)c1ccccc1', 'CCNC(=O)C(C)N(Cc1cccc(Cl)c1)C(=O)CN(c1cccc(Cl)c1Cl)S(=O)(=O)c1ccc(C)cc1', 'CCOc1c(OC)cccc1C1C(C(=O)OCCOC)=C(C)N=c2sc(=Cc3c(OCc4ccc(Cl)cc4)ccc4ccccc34)c(=O)n21', 'CC(C)CNC(=O)C(Cc1ccccc1)N(Cc1c(Cl)cccc1Cl)C(=O)CCCN1c2cccc3cccc(c23)S1(=O)=O', 'C#CC(O)=CC1=C(Cn2c3ccccc3c3ccccc32)N(c2nc(-c3ccccc3)nc(-c3ccccc3)n2)C2C(O)=C(O)C(O)=C12', 'COc1c(O)c2c(c(O)c1OC)C(O)(O)C(O)(C(O)(O)C1(O)C(O)(O)C(O)(O)N(C(O)(O)c3ccccc3)C(O)(O)C1(O)O)C2=O']

train_dataframe = train_dataframe[~train_dataframe.SMILES_nostereo.isin(remove_groups)].sort_values(by = ['SMILES_nostereo', 'ID']).reset_index(drop = True)
val_dataframe = val_dataframe[~val_dataframe.SMILES_nostereo.isin(remove_groups)].sort_values(by = ['SMILES_nostereo', 'ID']).reset_index(drop = True)


if params['sample_1conformer'] == True:
    train_dataframe = train_dataframe.groupby('ID').sample(1, random_state = seed).sort_values('SMILES_nostereo').reset_index(drop = True)
    val_dataframe = val_dataframe.groupby('ID').sample(1, random_state = seed).sort_values('SMILES_nostereo').reset_index(drop = True)

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
                                    regression = 'RS_label_binary', # top_score, RS_label_binary, sign_rotation
                              )

val_dataset = Dataset_3D_GNN(val_dataframe, 
                                    regression = 'RS_label_binary', # top_score, RS_label_binary, sign_rotation
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
test_dataframe = pd.read_pickle(params['test_datafile'])
test_dataframe = test_dataframe[~test_dataframe.SMILES_nostereo.isin(remove_groups)].sort_values(by = ['SMILES_nostereo', 'ID']).reset_index(drop = True)

test_dataset = Dataset_3D_GNN(test_dataframe, 
                                    regression = 'RS_label_binary', # top_score, RS_label_binary, sign_rotation
                              )

test_loader = torch_geometric.data.DataLoader(test_dataset, num_workers = num_workers, batch_size = 1000, shuffle = False)

targets, outputs = evaluate_classification_loop(model, test_loader, device, batch_size = 1000, dataset_size = len(test_dataset))

results_df = deepcopy(test_dataframe[['ID', 'SMILES_nostereo', 'RS_label_binary']])
results_df['targets'] = targets
results_df['outputs'] = outputs

results_df.to_csv(PATH + 'best_model_test_results.csv')

print('completed processes')

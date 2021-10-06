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
from model.train_functions import binary_ranking_regression_loop_alpha
from model.train_models import train_binary_ranking_regression_model
from model.datasets_samplers import MaskedGraphDataset, StereoBatchSampler, SiameseBatchSampler, Sample_Map_To_Positives, Sample_Map_To_Negatives, NegativeBatchSampler, SingleConformerBatchSampler

import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
import hyperopt as hp
from ray.tune.suggest.hyperopt import HyperOptSearch

def trainable(config, checkpoint_dir = None, max_N_epochs = None, train_dataframe = None, val_dataframe = None):
    # config is a dictionary holding all "tune"-able hyperparameters
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    seed = 1
    np.random.seed(seed = seed)
    torch.manual_seed(seed)
    
    encoder_hidden_sizes = [config['encoder_hidden_size']]*config['encoder_hidden_layer_number']
    F_H = config['F_H']
    lr = config['lr']
    params = {
        "layers_dict":
        {
          "EConv_mlp_hidden_sizes":   [config['EConv_mlp_hidden_size']]*config['EConv_mlp_hidden_layer_number'], 
          "GAT_hidden_node_sizes":    [config['GAT_hidden_node_size']]*config['GAT_hidden_layer_number'], 
    
          "encoder_hidden_sizes_D":   encoder_hidden_sizes, 
          "encoder_hidden_sizes_phi": encoder_hidden_sizes, 
          "encoder_hidden_sizes_c": encoder_hidden_sizes,
          "encoder_hidden_sizes_alpha": encoder_hidden_sizes,
            
          "encoder_hidden_sizes_sinusoidal_shift": [config['encoder_sinusoidal_shift_hidden_size']] * config['encoder_sinusoidal_shift_hidden_layer_number'], 
          "output_mlp_hidden_sizes": [config['output_mlp_hidden_size']]*config['output_mlp_hidden_layer_number']
        },
    
    
        "activation_dict":  
       {
            "encoder_hidden_activation_D": "torch.nn.LeakyReLU(negative_slope=0.01)", 
            "encoder_hidden_activation_phi": "torch.nn.LeakyReLU(negative_slope=0.01)", 
            "encoder_hidden_activation_c": "torch.nn.LeakyReLU(negative_slope=0.01)", 
            "encoder_hidden_activation_alpha": "torch.nn.LeakyReLU(negative_slope=0.01)",
            "encoder_hidden_activation_sinusoidal_shift": "torch.nn.LeakyReLU(negative_slope=0.01)",
            
            "encoder_output_activation_D": "torch.nn.Identity()", 
            "encoder_output_activation_phi": "torch.nn.Identity()", 
            "encoder_output_activation_c": "torch.nn.Identity()",
            "encoder_output_activation_alpha": "torch.nn.Identity()",
            "encoder_output_activation_sinusoidal_shift": "torch.nn.Identity()",
            
            "EConv_mlp_hidden_activation": "torch.nn.LeakyReLU(negative_slope=0.01)",
            "EConv_mlp_output_activation": "torch.nn.Identity()",
            
            "output_mlp_hidden_activation": "torch.nn.LeakyReLU(negative_slope=0.01)",
            "output_mlp_output_activation": "torch.nn.Identity()"
        },
    
        "pretrained": "",
    
        "F_z_list": [config['F_z']]*3,
        "F_H": F_H,
        "F_H_EConv": F_H,
        "GAT_N_heads": config['GAT_N_heads'],
        "EConv_bias": True,
        "GAT_bias": True,
        "encoder_biases": True,
        "dropout": 0.0,
        
        "chiral_message_passing": True,
        "CMP_EConv_MLP_hidden_sizes": [config['CMP_EConv_MLP_hidden_size']]*config['CMP_EConv_MLP_hidden_layer_number'], 
        "CMP_GAT_N_layers": config['CMP_GAT_N_layers'],
        "CMP_GAT_N_heads": config['CMP_GAT_N_heads'],
        
        "c_coefficient_mode": "learned",
        "c_coefficient_normalization": "sigmoid",
        
        "auxillary_torsion_loss": config['auxillary_torsion_loss'],
        
        "encoder_reduction": "sum",
        
        "output_concatenation_mode": "both",
            
        "default_lr": lr,
    
        "num_workers": 6,
        "batch_size": config['batch_size'],
        "N_epochs": max_N_epochs,
        
        "train_datafile": None,
        "validation_datafile": None,

        "iteration_mode": "stereoisomers",
        "sample_1conformer": False,
        "select_N_enantiomers": None,
        
        "mask_coordinates": False,
        "stereoMask": True,
        
        "grouping": "none",
        "weighted_sum": True,
    
        "stratified": False,
        "withoutReplacement": True,
    
        "loss_function": "MSE_MarginRankingLoss",
        "absolute_penalty": 1.0,
        "relative_penalty": 0.0,
        "ranking_margin": 0.3,
        
        "contrastive_vector": "none",
        "margin": None,
        
        "N_neg": 1,
        "N_pos": 0,
            
        "save": True
    }
    
    
    #CREATE MODEL
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
        F_H_embed = num_node_features, # dimension of initial node feature vector, currently 52
        F_E_embed = num_edge_features, # dimension of initial edge feature vector, currently 14
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
    
    # DEFINE OPTIMIZERS AND SCHEDULERS
    lr = params['default_lr']
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr = lr)
    optimizers = [optimizer]
    
    # Choosing Loss
    loss_function = params['loss_function']
    auxillary_torsion_loss = params['auxillary_torsion_loss']
    
    # only for contrastive learning
    margin = params['margin']
    contrastive_vector = params['contrastive_vector']
    
    # only for docking
    absolute_penalty = params['absolute_penalty'] # default is 1.0
    relative_penalty = params['relative_penalty'] # default is None (null). If a float >=0.0, we have to use a SiameseBatchSampler
    ranking_margin = params['ranking_margin']
    
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
                                        regression = 'top_score', # top_score, RS_label_binary, sign_rotation
                                        stereoMask = stereoMask,
                                        mask_coordinates = params['mask_coordinates'], 
                                        )
    
    val_dataset = MaskedGraphDataset(val_dataframe, 
                                        regression = 'top_score', # top_score, RS_label_binary, sign_rotation
                                        stereoMask = stereoMask,
                                        mask_coordinates = params['mask_coordinates'], 
                                        )
    
    num_workers = params['num_workers']
    train_loader = torch_geometric.data.DataLoader(train_dataset, batch_sampler = BatchSampler_train, num_workers = num_workers)
    val_loader = torch_geometric.data.DataLoader(val_dataset, batch_sampler = BatchSampler_val, num_workers = num_workers)
    
    
    # BEGIN TRAINING
    weighted_sum = params['weighted_sum'] # only for StereoBatchSampler
    N_epochs = params['N_epochs']    
    
    train_epoch_losses = []
    train_epoch_aux_losses = []
    train_epoch_abs_losses = []
    train_epoch_rel_losses = []
    train_epoch_accuracies = []
    
    val_epoch_losses = []
    val_epoch_aux_losses = []
    val_epoch_abs_losses = []
    val_epoch_rel_losses = []
    val_epoch_accuracies = []


    for epoch in tqdm(range(1, N_epochs+1)):
    
        train_losses, train_aux_losses, train_batch_sizes, train_abs_losses, train_rel_losses, train_accuracies = binary_ranking_regression_loop_alpha(model, train_loader, optimizers, device, epoch, batch_size, training = True, absolute_penalty = absolute_penalty, relative_penalty = relative_penalty, ranking_margin = ranking_margin, auxillary_torsion_loss = auxillary_torsion_loss)

        if weighted_sum:
            epoch_loss = torch.sum(torch.tensor(train_losses) * torch.tensor(train_batch_sizes)) / (torch.sum(torch.tensor(train_batch_sizes))) #weighted mean based on the batch sizes
            epoch_abs_loss = torch.sum(torch.tensor(train_abs_losses) * torch.tensor(train_batch_sizes)) / (torch.sum(torch.tensor(train_batch_sizes)))
            epoch_rel_loss = torch.sum(torch.tensor(train_rel_losses) * torch.tensor(train_batch_sizes)) / (torch.sum(torch.tensor(train_batch_sizes)))
            epoch_aux_loss = torch.sum(torch.tensor(train_aux_losses) * torch.tensor(train_batch_sizes)) / (torch.sum(torch.tensor(train_batch_sizes)))
            epoch_acc = torch.sum(torch.tensor(train_accuracies) * torch.tensor(train_batch_sizes)) / (torch.sum(torch.tensor(train_batch_sizes)))
        else:
            epoch_loss = torch.mean(torch.tensor(train_losses))
            epoch_abs_loss = torch.mean(torch.tensor(train_abs_losses))
            epoch_rel_loss = torch.mean(torch.tensor(train_rel_losses))
            epoch_aux_loss = torch.mean(torch.tensor(train_aux_losses))
            epoch_acc = torch.mean(torch.tensor(train_accuracies))
            
        train_epoch_losses.append(epoch_loss)
        train_epoch_abs_losses.append(epoch_abs_loss)
        train_epoch_rel_losses.append(epoch_rel_loss)
        train_epoch_aux_losses.append(epoch_aux_loss)
        train_epoch_accuracies.append(epoch_acc)
        
        with torch.no_grad():
            val_losses, val_aux_losses, val_batch_sizes, val_abs_losses, val_rel_losses, val_accuracies = binary_ranking_regression_loop_alpha(model, val_loader, optimizers, device, epoch, batch_size, training = False, absolute_penalty = absolute_penalty, relative_penalty = relative_penalty, ranking_margin = ranking_margin, auxillary_torsion_loss = auxillary_torsion_loss)
        
            if weighted_sum:
                val_epoch_loss = torch.sum(torch.tensor(val_losses) * torch.tensor(val_batch_sizes)) / (torch.sum(torch.tensor(val_batch_sizes))) #weighted mean based on the batch sizes
                val_epoch_abs_loss = torch.sum(torch.tensor(val_abs_losses) * torch.tensor(val_batch_sizes)) / (torch.sum(torch.tensor(val_batch_sizes)))
                val_epoch_rel_loss = torch.sum(torch.tensor(val_rel_losses) * torch.tensor(val_batch_sizes)) / (torch.sum(torch.tensor(val_batch_sizes)))
                val_epoch_aux_loss = torch.sum(torch.tensor(val_aux_losses) * torch.tensor(val_batch_sizes)) / (torch.sum(torch.tensor(val_batch_sizes)))
                val_epoch_acc = torch.sum(torch.tensor(val_accuracies) * torch.tensor(val_batch_sizes)) / (torch.sum(torch.tensor(val_batch_sizes)))
            else:
                val_epoch_loss = torch.mean(torch.tensor(val_losses))
                val_epoch_abs_loss = torch.mean(torch.tensor(val_abs_losses))
                val_epoch_rel_loss = torch.mean(torch.tensor(val_rel_losses))
                val_epoch_aux_loss = torch.mean(torch.tensor(val_aux_losses))
                val_epoch_acc = torch.mean(torch.tensor(val_accuracies))
                
            val_epoch_losses.append(val_epoch_loss)
            val_epoch_abs_losses.append(val_epoch_abs_loss)
            val_epoch_rel_losses.append(val_epoch_rel_loss)
            val_epoch_aux_losses.append(val_epoch_aux_loss)
            val_epoch_accuracies.append(val_epoch_acc)
            
        #reporting to raytune and checkpointing
        if epoch >= 5:
            if epoch % 5 == 0:
                print('checkpointing...')
                with tune.checkpoint_dir(step=epoch) as checkpoint_dir:
                    path = os.path.join(checkpoint_dir, "checkpoint")
                    torch.save(model.state_dict(), path + '_model.pt')
                    
                    torch.save(torch.tensor(train_epoch_losses), path + '_train_losses.pt')
                    torch.save(torch.tensor(train_epoch_abs_losses), path + '_train_abs_losses.pt')
                    torch.save(torch.tensor(train_epoch_rel_losses), path + '_train_rel_losses.pt')
                    
                    torch.save(torch.tensor(val_epoch_losses), path + '_validation_losses.pt')
                    torch.save(torch.tensor(val_epoch_abs_losses), path + '_validation_abs_losses.pt')
                    torch.save(torch.tensor(val_epoch_rel_losses), path + '_validation_rel_losses.pt')
                    
                    torch.save(torch.tensor(train_epoch_aux_losses), path + '_train_aux_losses.pt')
                    torch.save(torch.tensor(val_epoch_aux_losses), path + '_validation_aux_losses.pt')
                    
                    torch.save(torch.tensor(train_epoch_accuracies), path + '_train_accuracies.pt')
                    torch.save(torch.tensor(val_epoch_accuracies), path + '_validation_accuracies.pt')

        print('reporting results to tune...')
        tune.report(accuracy = val_epoch_acc, loss = val_epoch_loss, abs_loss = val_epoch_abs_loss, rel_loss = val_epoch_rel_loss, aux_loss = val_epoch_aux_loss)

    print('completed training')
    
    
def main():
    # initialized ray cluster with resources specified in batch submission script
    ray.init(
        address=os.environ["ip_head"],
        _node_ip_address=os.environ["ip_head"].split(":")[0])
    print('Ray cluster online with resources:')
    print(ray.cluster_resources())
    
    config = {
        "EConv_mlp_hidden_size": tune.choice([32, 64, 128, 256]),
        "EConv_mlp_hidden_layer_number": tune.choice([1, 2]),
    
        "GAT_hidden_node_size": tune.choice([16, 32, 64]),
        "GAT_hidden_layer_number": tune.choice([1, 2, 3]),
    
        "encoder_hidden_size": tune.choice([32, 64, 128, 256]),
        "encoder_hidden_layer_number": tune.choice([1, 2, 3, 4]),
    
        "encoder_sinusoidal_shift_hidden_size": tune.choice([32, 64, 128, 256]),
        "encoder_sinusoidal_shift_hidden_layer_number": tune.choice([1, 2, 3, 4]),
    
        "output_mlp_hidden_size": tune.choice([32, 64, 128, 256]),
        "output_mlp_hidden_layer_number": tune.choice([1, 2, 3, 4]),
    
        "F_z": tune.choice([8, 16, 32, 64]),
        "F_H": tune.choice([8, 16, 32, 64]),
    
        "GAT_N_heads": tune.choice([1, 2, 4, 8]),
    
        "CMP_EConv_MLP_hidden_size": tune.choice([32, 64, 128, 256]),
        "CMP_EConv_MLP_hidden_layer_number": tune.choice([1, 2, 3, 4]),
    
        "CMP_GAT_N_layers": tune.choice([1, 2, 3, 4]),
        "CMP_GAT_N_heads": tune.choice([1, 2, 4, 8]),
    
        "auxillary_torsion_loss": tune.loguniform(1e-4, 1e-2),
    
        "lr": tune.loguniform(5e-5, 5e-3),
        "batch_size": tune.choice([16, 32, 64, 128, 256])
    }
    
    configs_to_evaluate = [
        {
        "EConv_mlp_hidden_size": 128,
        "EConv_mlp_hidden_layer_number": 2,

        "GAT_hidden_node_size": 16,
        "GAT_hidden_layer_number": 2,

        "encoder_hidden_size": 256,
        "encoder_hidden_layer_number": 2,

        "encoder_sinusoidal_shift_hidden_size": 256,
        "encoder_sinusoidal_shift_hidden_layer_number": 2,

        "output_mlp_hidden_size": 256,
        "output_mlp_hidden_layer_number": 2,

        "F_z": 16,
        "F_H": 16,

        "GAT_N_heads": 4,

        "CMP_EConv_MLP_hidden_size": 256,
        "CMP_EConv_MLP_hidden_layer_number": 2,

        "CMP_GAT_N_layers": 2,
        "CMP_GAT_N_heads": 4,

        "auxillary_torsion_loss": 0.0001,

        "lr": 0.00035,
        "batch_size": 32,

        }
    ]
    
    # READING DATA
    
    # using training and validation splits
    train_dataframe = pd.read_pickle('final_data_splits/train_small_enantiomers_stable_full_screen_docking_MOL_margin3_234622_48384_24192.pkl')
    
    val_dataframe = pd.read_pickle('final_data_splits/validation_small_enantiomers_stable_full_screen_docking_MOL_margin3_49878_10368_5184.pkl')
    
    np.random.seed(0)
    random.seed(0)
    torch.manual_seed(0)

    max_N_epochs = 50 # 50
    
    tune_scheduler = ASHAScheduler(
            metric="accuracy",
            mode="max",
            max_t=max_N_epochs,
            grace_period=5,
            reduction_factor=3,
            brackets = 1)
    
    reporter = CLIReporter(
            metric_columns=["accuracy", "loss", "abs_loss", "rel_loss", "aux_loss", "training_iteration"]
            )
    
    hyperopt_search = HyperOptSearch(
        metric="accuracy", 
        mode="max",
        points_to_evaluate=configs_to_evaluate
        )
    
    result = tune.run(
            tune.with_parameters(trainable, max_N_epochs = max_N_epochs, train_dataframe = train_dataframe, val_dataframe = val_dataframe),
            resources_per_trial={"cpu": 10}, # resources_per_trial={"cpu": 12, "gpu": 1}
            config=config,
            num_samples=100, # number of trials / sets of hyperparameters # 50
            scheduler=tune_scheduler,
            search_alg = hyperopt_search,
            progress_reporter=reporter,
            local_dir = './raytune_results'
            )
    
if __name__ == '__main__':
    main()

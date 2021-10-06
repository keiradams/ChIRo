import torch
import torch.nn as nn
import torch_geometric
import datetime
import numpy as np
from tqdm import tqdm
import math
from collections import OrderedDict

from .optimization_functions import BCE_loss, tripletLoss, MSE

from itertools import chain

import random

def compute_pnorm(parameters):
    return math.sqrt(sum([p.norm().item() ** 2 for p in parameters]))

def compute_gnorm(parameters):
    return math.sqrt(sum([p.grad.norm().item() ** 2 for p in parameters if p.grad is not None]))

def get_local_structure_map(psi_indices):
    LS_dict = OrderedDict()
    LS_map = torch.zeros(psi_indices.shape[1], dtype = torch.long)
    v = 0
    for i, indices in enumerate(psi_indices.T):
        tupl = (int(indices[1]), int(indices[2]))
        if tupl not in LS_dict:
            LS_dict[tupl] = v
            v += 1
        LS_map[i] = LS_dict[tupl]

    alpha_indices = torch.zeros((2, len(LS_dict)), dtype = torch.long)
    for i, tupl in enumerate(LS_dict):
        alpha_indices[:,i] = torch.LongTensor(tupl)

    return LS_map, alpha_indices

def binary_ranking_regression_loop_alpha(model, loader, optimizers, device, epoch, batch_size, training = True, absolute_penalty = 1.0, relative_penalty = 0.0, ranking_margin = 0.3, auxillary_torsion_loss = 0.02):
    if training:
        model.train()
    else:
        model.eval()

    batch_losses = []
    batch_aux_losses = []
    batch_rel_losses = []
    batch_abs_losses = []
    batch_sizes = []
    
    batch_acc = []
    
    for batch in loader:
        batch_data, y = batch

        psi_indices = batch_data.dihedral_angle_index
        LS_map, alpha_indices = get_local_structure_map(psi_indices)

        batch_data = batch_data.to(device)
        LS_map = LS_map.to(device)
        alpha_indices = alpha_indices.to(device)
        y = y.to(device)

        if training:
            for opt in optimizers:
                opt.zero_grad()
        
        output, latent_vector, phase_shift_norm, z_alpha, mol_embedding, c_tensor, phase_cos, phase_sin, sin_cos_psi, sin_cos_alpha = model(batch_data, LS_map, alpha_indices)

        loss_absolute = MSE(y.squeeze(), output.squeeze()) # plain MSE loss
        
        criterion = torch.nn.MarginRankingLoss(margin=ranking_margin)
        
        #used in conjunction with negative batch sampler, where the negative immediately follows each anchor
        # notice that we treat the less negative score as being ranked "higher"
        loss_relative = criterion(output[0::2].squeeze(), output[1::2].squeeze(), torch.sign((y[0::2].squeeze() - y[1::2].squeeze()) + 1e-8).squeeze())
        
        aux_loss = torch.mean(torch.abs(1.0 - phase_shift_norm.squeeze()))
        
        loss = (loss_relative*relative_penalty) + (loss_absolute*absolute_penalty)
        backprop_loss = loss + aux_loss*auxillary_torsion_loss
        
        if training:
            backprop_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10, norm_type=2)
        
            for opt in optimizers:
                opt.step()
        
        # return  (binary) ranking accuracies, using margin = 0.1 and equivalence = 0.0
        target_ranking = ((torch.round(y[0::2].squeeze() * 100.) / 100.) > (torch.round(y[1::2].squeeze() * 100.) / 100.)).type(torch.float)
        output_ranking = ((torch.round(output[0::2].squeeze() * 100.) / 100.) > (torch.round(output[1::2].squeeze() * 100.) / 100.)).type(torch.float)
        top_1_acc = torch.sum(output_ranking == target_ranking) / float(output_ranking.shape[0])
        
        batch_acc.append(top_1_acc.item())
        
        batch_sizes.append(y.shape[0])
        batch_losses.append(loss.item())
        batch_aux_losses.append(aux_loss.item())
        
        batch_rel_losses.append(loss_relative.item())
        batch_abs_losses.append(loss_absolute.item())
        
    return batch_losses, batch_aux_losses, batch_sizes, batch_abs_losses, batch_rel_losses, batch_acc

def evaluate_binary_ranking_regression_loop_alpha(model, loader, device, batch_size, dataset_size):
    model.eval()
    
    all_targets = torch.zeros(dataset_size).to(device)
    all_outputs = torch.zeros(dataset_size).to(device)
    
    start = 0
    for batch in tqdm(loader):
        batch_data, y = batch

        psi_indices = batch_data.dihedral_angle_index
        LS_map, alpha_indices = get_local_structure_map(psi_indices)

        batch_data = batch_data.to(device)
        LS_map = LS_map.to(device)
        alpha_indices = alpha_indices.to(device)
        y = y.to(device)
        
        with torch.no_grad():
            output, latent_vector, phase_shift_norm, z_alpha, mol_embedding, c_tensor, phase_cos, phase_sin, sin_cos_psi, sin_cos_alpha = model(batch_data, LS_map, alpha_indices)
        
            all_targets[start:start + y.squeeze().shape[0]] = y.squeeze()
            all_outputs[start:start + y.squeeze().shape[0]] = output.squeeze()
            start += y.squeeze().shape[0]
       
    return all_targets.detach().cpu().numpy(), all_outputs.detach().cpu().numpy()


def classification_loop_alpha(model, loader, optimizers, device, epoch, batch_size, training = True, auxillary_torsion_loss = 0.02):
    if training:
        model.train()
    else:
        model.eval()

    batch_losses = []
    batch_aux_losses = []
    batch_sizes = []
    batch_accuracies = []
    
    for batch in loader:
        batch_data, y = batch
        y = y.type(torch.float32)
        
        psi_indices = batch_data.dihedral_angle_index
        LS_map, alpha_indices = get_local_structure_map(psi_indices)

        batch_data = batch_data.to(device)
        LS_map = LS_map.to(device)
        alpha_indices = alpha_indices.to(device)
        y = y.to(device)

        if training:
            for opt in optimizers:
                opt.zero_grad()
        
        output, latent_vector, phase_shift_norm, z_alpha, mol_embedding, c_tensor, phase_cos, phase_sin, sin_cos_psi, sin_cos_alpha = model(batch_data, LS_map, alpha_indices)
        
        aux_loss = torch.mean(torch.abs(1.0 - phase_shift_norm.squeeze()))
        loss = BCE_loss(y.squeeze(), output.squeeze())
        backprop_loss = loss + aux_loss*auxillary_torsion_loss
        
        acc = 1.0 - (torch.sum(torch.abs(y.squeeze().detach() - torch.round(torch.sigmoid(output.squeeze().detach())))) / y.shape[0])
        
        if training:
            backprop_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10, norm_type=2)
        
            for opt in optimizers:
                opt.step()
        
        batch_sizes.append(y.shape[0])
        batch_losses.append(loss.item())
        batch_aux_losses.append(aux_loss.item())
        batch_accuracies.append(acc.item())
        
    return batch_losses, batch_aux_losses, batch_sizes, batch_accuracies


def evaluate_classification_loop_alpha(model, loader, device, batch_size, dataset_size):
    model.eval()
    
    all_targets = torch.zeros(dataset_size).to(device)
    all_outputs = torch.zeros(dataset_size).to(device)
    
    start = 0
    for batch in loader:
        batch_data, y = batch
        y = y.type(torch.float32)
        
        psi_indices = batch_data.dihedral_angle_index
        LS_map, alpha_indices = get_local_structure_map(psi_indices)

        batch_data = batch_data.to(device)
        LS_map = LS_map.to(device)
        alpha_indices = alpha_indices.to(device)
        y = y.to(device) 

        with torch.no_grad():
            output, latent_vector, phase_shift_norm, z_alpha, mol_embedding, c_tensor, phase_cos, phase_sin, sin_cos_psi, sin_cos_alpha = model(batch_data, LS_map, alpha_indices)
            
            all_targets[start:start + y.squeeze().shape[0]] = y.squeeze()
            all_outputs[start:start + y.squeeze().shape[0]] = output.squeeze()
            start += y.squeeze().shape[0]
       
    return all_targets.detach().cpu().numpy(), all_outputs.detach().cpu().numpy()


def contrastive_loop_alpha(model, loader, optimizers, device, epoch, loss_function, batch_size, margin, training = True, contrastive_vector = 'z_alpha', auxillary_torsion_loss = 0.02):
    if training:
        model.train()
    else:
        model.eval()

    batch_losses = []
    batch_aux_losses = []
    
    for batch_data in loader:
        
        psi_indices = batch_data.dihedral_angle_index
        LS_map, alpha_indices = get_local_structure_map(psi_indices)

        batch_data = batch_data.to(device)
        LS_map = LS_map.to(device)
        alpha_indices = alpha_indices.to(device)

        if training:
            for opt in optimizers:
                opt.zero_grad()
        
        latent_vector, phase_shift_norm, z_alpha, mol_embedding, c_tensor, phase_cos, phase_sin, sin_cos_psi, sin_cos_alpha = model(batch_data, LS_map, alpha_indices)
        
        aux_loss = torch.mean(torch.abs(1.0 - phase_shift_norm.squeeze()))

        if contrastive_vector == 'z_alpha':
            #choosing only torsion latent vectors (final third, if concatenated)
            anchor = latent_vector[0::3, (latent_vector.shape[1]//3) * 2 :]
            positive = latent_vector[1::3, (latent_vector.shape[1]//3) * 2 :]
            negative = latent_vector[2::3, (latent_vector.shape[1]//3) * 2 :]
        elif contrastive_vector == 'conformer':
            anchor = latent_vector[0::3, :]
            positive = latent_vector[1::3, :]
            negative = latent_vector[2::3, :]
        elif contrastive_vector == 'molecule':
            anchor = mol_embedding[0::3, :]
            positive = mol_embedding[1::3, :]
            negative = mol_embedding[2::3, :]
        elif contrastive_vector == 'both':
            embedding = torch.cat((mol_embedding, latent_vector), dim = 1)
            anchor = embedding[0::3, :]
            positive = embedding[1::3, :]
            negative = embedding[2::3, :]
            
        if loss_function == 'euclidean':
            loss = tripletLoss(anchor, positive, negative, margin = margin, reduction = 'mean', distance_metric = 'euclidean')
        elif loss_function == 'euclidean-normalized':
            loss = tripletLoss(anchor, positive, negative, margin = margin, reduction = 'mean', distance_metric = 'euclidean_normalized')
        elif loss_function == 'manhattan':
            loss = tripletLoss(anchor, positive, negative, margin = margin, reduction = 'mean', distance_metric = 'manhattan')
        elif loss_function == 'cosine':
            loss = tripletLoss(anchor, positive, negative, margin = margin, reduction = 'mean', distance_metric = 'cosine')
        
        backprop_loss = loss + aux_loss*auxillary_torsion_loss
        if training:
            backprop_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10, norm_type=2)
        
            for opt in optimizers:
                opt.step()
        
        batch_losses.append(loss.item())
        batch_aux_losses.append(aux_loss.item())
        
    return batch_losses, batch_aux_losses

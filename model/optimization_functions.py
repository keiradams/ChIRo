import torch
import torch.nn as nn
import numpy as np
import math
import torch_scatter

def BCE_loss(y, y_hat):
    BCE = torch.nn.BCEWithLogitsLoss()
    return BCE(y_hat, y)

def MSE(y, y_hat):
    MSE = torch.mean(torch.square(y - y_hat))
    return MSE

def tripletLoss(z_anchor, z_positive, z_negative, margin = 1.0, reduction = 'mean', distance_metric = 'euclidean'):
    if distance_metric == 'euclidean':
        criterion = torch.nn.TripletMarginWithDistanceLoss(distance_function=torch.nn.PairwiseDistance(p=2.0), 
            margin=margin, 
            swap=False, 
            reduction=reduction)
    elif distance_metric == 'euclidean_normalized':
        criterion = torch.nn.TripletMarginWithDistanceLoss(distance_function=torch.nn.PairwiseDistance(p=2.0), 
            margin=margin, 
            swap=False, 
            reduction=reduction)
    elif distance_metric == 'manhattan':
        criterion = torch.nn.TripletMarginWithDistanceLoss(distance_function=torch.nn.PairwiseDistance(p=1.0), 
            margin=margin, 
            swap=False, 
            reduction=reduction)
    elif distance_metric == 'cosine':
        criterion = torch.nn.TripletMarginWithDistanceLoss(distance_function= lambda x, y: 1.0 - torch.nn.functional.cosine_similarity(x, y),  
            margin=margin, 
            swap=False, 
            reduction=reduction)
    else:
        raise Exception(f'distance metric {distance_metric} is not implemented')

    if distance_metric == 'euclidean_normalized':
        z_anchor = z_anchor / torch.linalg.norm(z_anchor + 1e-10, dim=1, keepdim = True)
        z_positive = z_positive / torch.linalg.norm(z_positive + 1e-10, dim=1, keepdim = True)
        z_negative = z_negative / torch.linalg.norm(z_negative + 1e-10, dim=1, keepdim = True)

    loss = criterion(z_anchor, z_positive, z_negative)
    return loss

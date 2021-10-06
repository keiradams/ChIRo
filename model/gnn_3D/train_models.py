import torch
import torch.nn as nn
import torch_geometric
import datetime
import numpy as np
from tqdm import tqdm
from copy import deepcopy

from .train_functions import classification_loop, contrastive_loop, binary_ranking_regression_loop

def train_binary_ranking_regression_model(model, train_loader, val_loader, N_epochs, optimizer, device, batch_size, absolute_penalty = 1.0, relative_penalty = 0.0, ranking_margin = 0.3, weighted_sum = False, save = True, PATH = ''):
    train_epoch_losses = []
    train_epoch_abs_losses = []
    train_epoch_rel_losses = []
    train_epoch_accuracies = []
    
    val_epoch_losses = []
    val_epoch_abs_losses = []
    val_epoch_rel_losses = []
    val_epoch_accuracies = []

    best_val_acc = 0.0
    best_val_loss = np.inf
    best_epoch = 0
    best_state_dict = {}
    
    for epoch in tqdm(range(1, N_epochs+1)):
    
        train_losses, train_batch_sizes, train_abs_losses, train_rel_losses, train_accuracies = binary_ranking_regression_loop(model, train_loader, optimizer, device, epoch, batch_size, training = True, absolute_penalty = absolute_penalty, relative_penalty = relative_penalty, ranking_margin = ranking_margin)

        if weighted_sum:
            epoch_loss = torch.sum(torch.tensor(train_losses) * torch.tensor(train_batch_sizes)) / (torch.sum(torch.tensor(train_batch_sizes))) #weighted mean based on the batch sizes
            epoch_abs_loss = torch.sum(torch.tensor(train_abs_losses) * torch.tensor(train_batch_sizes)) / (torch.sum(torch.tensor(train_batch_sizes)))
            epoch_rel_loss = torch.sum(torch.tensor(train_rel_losses) * torch.tensor(train_batch_sizes)) / (torch.sum(torch.tensor(train_batch_sizes)))
            epoch_acc = torch.sum(torch.tensor(train_accuracies) * torch.tensor(train_batch_sizes)) / (torch.sum(torch.tensor(train_batch_sizes)))
        else:
            epoch_loss = torch.mean(torch.tensor(train_losses))
            epoch_abs_loss = torch.mean(torch.tensor(train_abs_losses))
            epoch_rel_loss = torch.mean(torch.tensor(train_rel_losses))
            epoch_acc = torch.mean(torch.tensor(train_accuracies))
            
        train_epoch_losses.append(epoch_loss)
        train_epoch_abs_losses.append(epoch_abs_loss)
        train_epoch_rel_losses.append(epoch_rel_loss)
        train_epoch_accuracies.append(epoch_acc)
        
        with torch.no_grad():
            val_losses, val_batch_sizes, val_abs_losses, val_rel_losses, val_accuracies = binary_ranking_regression_loop(model, val_loader, optimizer, device, epoch, batch_size, training = False, absolute_penalty = absolute_penalty, relative_penalty = relative_penalty, ranking_margin = ranking_margin)
        
            if weighted_sum:
                val_epoch_loss = torch.sum(torch.tensor(val_losses) * torch.tensor(val_batch_sizes)) / (torch.sum(torch.tensor(val_batch_sizes))) #weighted mean based on the batch sizes
                val_epoch_abs_loss = torch.sum(torch.tensor(val_abs_losses) * torch.tensor(val_batch_sizes)) / (torch.sum(torch.tensor(val_batch_sizes)))
                val_epoch_rel_loss = torch.sum(torch.tensor(val_rel_losses) * torch.tensor(val_batch_sizes)) / (torch.sum(torch.tensor(val_batch_sizes)))
                val_epoch_acc = torch.sum(torch.tensor(val_accuracies) * torch.tensor(val_batch_sizes)) / (torch.sum(torch.tensor(val_batch_sizes)))
            else:
                val_epoch_loss = torch.mean(torch.tensor(val_losses))
                val_epoch_abs_loss = torch.mean(torch.tensor(val_abs_losses))
                val_epoch_rel_loss = torch.mean(torch.tensor(val_rel_losses))
                val_epoch_acc = torch.mean(torch.tensor(val_accuracies))
                
            val_epoch_losses.append(val_epoch_loss)
            val_epoch_abs_losses.append(val_epoch_abs_loss)
            val_epoch_rel_losses.append(val_epoch_rel_loss)
            val_epoch_accuracies.append(val_epoch_acc)
            
            if val_epoch_acc > best_val_acc:
                best_val_acc = val_epoch_acc
                best_epoch = epoch
                best_state_dict = deepcopy(model.state_dict())
                if save == True:
                    torch.save(model.state_dict(), PATH + 'best_model.pt')
                    print('\n    saving best model:' + str(epoch))
                    print('    Best Epoch:', epoch, 'Train Loss:', epoch_loss, 'Train Acc.:', epoch_acc,'Validation Loss:', val_epoch_loss, 'Validation Acc.:', val_epoch_acc)
                    print('        Train Losses (abs., rel.):', (epoch_abs_loss, epoch_rel_loss), 'Validation Losses (abs., rel.):', (val_epoch_abs_loss, val_epoch_rel_loss))

            if epoch % 5 == 0:
                print('Epoch:', epoch, 'Train Loss:', epoch_loss, 'Train Acc.:', epoch_acc,'Validation Loss:', val_epoch_loss, 'Validation Acc.:', val_epoch_acc)
                print('        Train Losses (abs., rel.):', (epoch_abs_loss, epoch_rel_loss), 'Validation Losses (abs., rel.):', (val_epoch_abs_loss, val_epoch_rel_loss))
                if (save == True) and (epoch % 5 == 0):
                    torch.save(model.state_dict(), PATH + 'checkpoint_models/' + 'checkpoint_model_' + str(epoch) + '.pt')
                    torch.save(train_epoch_losses, PATH + 'train_epoch_losses.pt')
                    torch.save(train_epoch_abs_losses, PATH + 'train_epoch_abs_losses.pt')
                    torch.save(train_epoch_rel_losses, PATH + 'train_epoch_rel_losses.pt')
                    
                    torch.save(val_epoch_losses, PATH + 'val_epoch_losses.pt')
                    torch.save(val_epoch_abs_losses, PATH + 'val_epoch_abs_losses.pt')
                    torch.save(val_epoch_rel_losses, PATH + 'val_epoch_rel_losses.pt')
                                        
                    torch.save(train_epoch_accuracies, PATH + 'train_epoch_accuracies.pt')
                    torch.save(val_epoch_accuracies, PATH + 'val_epoch_accuracies.pt')

    return best_state_dict


def train_classification_model(model, train_loader, val_loader, N_epochs, optimizer, device, batch_size, weighted_sum = False, save = True, PATH = ''):
    
    train_epoch_losses = []
    train_epoch_accuracy = []
    
    val_epoch_losses = []
    val_epoch_accuracy = []
    
    best_val_accuracy = 0.0
    best_epoch = 0
    best_state_dict = {}
    
    for epoch in tqdm(range(1, N_epochs+1)):
    
        train_losses, train_batch_sizes, train_batch_accuracy = classification_loop(model, train_loader, optimizer, device, epoch, batch_size, training = True)

        if weighted_sum:
            epoch_loss = torch.sum(torch.tensor(train_losses) * torch.tensor(train_batch_sizes)) / (torch.sum(torch.tensor(train_batch_sizes))) #weighted mean based on the batch sizes
            train_accuracy = torch.sum(torch.tensor(train_batch_accuracy) * torch.tensor(train_batch_sizes)) / (torch.sum(torch.tensor(train_batch_sizes)))
        else:
            epoch_loss = torch.mean(torch.tensor(train_losses))
            train_accuracy = torch.mean(torch.tensor(train_batch_accuracy))

        train_epoch_losses.append(epoch_loss)
        train_epoch_accuracy.append(train_accuracy)

        with torch.no_grad():
            val_losses, val_batch_sizes, val_batch_accuracy = classification_loop(model, val_loader, optimizer, device, epoch, batch_size, training = False)
        
            if weighted_sum:
                val_epoch_loss = torch.sum(torch.tensor(val_losses) * torch.tensor(val_batch_sizes)) / (torch.sum(torch.tensor(val_batch_sizes))) #weighted mean based on the batch sizes
                val_accuracy = torch.sum(torch.tensor(val_batch_accuracy) * torch.tensor(val_batch_sizes)) / (torch.sum(torch.tensor(val_batch_sizes)))
            else:
                val_epoch_loss = torch.mean(torch.tensor(val_losses))
                val_accuracy = torch.mean(torch.tensor(val_batch_accuracy))

            val_epoch_losses.append(val_epoch_loss)
            val_epoch_accuracy.append(val_accuracy)
        
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_epoch = epoch
                best_state_dict = deepcopy(model.state_dict())
                if save == True:
                    torch.save(model.state_dict(), PATH + 'best_model.pt')
                    print('\n    saving best model:' + str(epoch))
                    print('    Best Epoch:', epoch, 'Train Loss:', epoch_loss, 'Validation Loss:', val_epoch_loss, 'Validation Acc.', val_accuracy)

            if epoch % 1 == 0:
                print('Epoch:', epoch, 'Train Loss:', epoch_loss, 'Validation Loss:', val_epoch_loss)
                print('    Epoch:', epoch, 'Train Loss:', epoch_loss, 'Validation Loss:', val_epoch_loss, 'Validation Acc.', val_accuracy)
                if (save == True) and (epoch % 5 == 0):
                    torch.save(model.state_dict(), PATH + 'checkpoint_models/' + 'checkpoint_model_' + str(epoch) + '.pt')
                    torch.save(train_epoch_losses, PATH + 'train_epoch_losses.pt')
                    torch.save(val_epoch_losses, PATH + 'val_epoch_losses.pt')
    
    return best_state_dict

def train_contrastive_model(model, train_loader, val_loader, N_epochs, optimizer, device, loss_function, batch_size, margin, save = True, PATH = ''):
    train_epoch_losses = []
    
    val_epoch_losses = []
    
    best_val_loss = np.inf
    best_epoch = 0
    best_state_dict = {}
    
    for epoch in tqdm(range(1, N_epochs+1)):
    
        train_losses = contrastive_loop(model, train_loader, optimizer, device, epoch, loss_function, batch_size, margin, training = True)

        epoch_loss = torch.mean(torch.tensor(train_losses))

        train_epoch_losses.append(epoch_loss)

        with torch.no_grad():
            val_losses = contrastive_loop(model, train_loader, optimizer, device, epoch, loss_function, batch_size, margin, training = False)
        
            val_epoch_loss = torch.mean(torch.tensor(val_losses))

            val_epoch_losses.append(val_epoch_loss)
        
            if val_epoch_loss < best_val_loss:
                best_val_loss = val_epoch_loss
                best_epoch = epoch
                best_state_dict = deepcopy(model.state_dict())
                if save == True:
                    torch.save(model.state_dict(), PATH + 'best_model.pt')
                    print('\n    saving best model:' + str(epoch))
                    print('    Best Epoch:', epoch, 'Train Loss:', epoch_loss, 'Validation Loss:', val_epoch_loss)

            if epoch % 1 == 0:
                print('Epoch:', epoch, 'Train Loss:', epoch_loss, 'Validation Loss:', val_epoch_loss)
                print('    Epoch:', epoch, 'Train Loss:', epoch_loss, 'Validation Loss:', val_epoch_loss)
                if (save == True) and (epoch % 5 == 0):
                    torch.save(model.state_dict(), PATH + 'checkpoint_models/' + 'checkpoint_model_' + str(epoch) + '.pt')
                    torch.save(train_epoch_losses, PATH + 'train_epoch_losses.pt')
                    torch.save(val_epoch_losses, PATH + 'val_epoch_losses.pt')
    
    return best_state_dict

import torch
import torch.nn as nn
import torch_geometric

import math
import pandas as pd
import numpy as np

from copy import deepcopy
from itertools import chain

import rdkit
from rdkit.Chem import rdMolTransforms
from rdkit.Chem import TorsionFingerprints
import networkx as nx

from tqdm import tqdm
import datetime
import random

from .embedding_functions import embedConformerWithAllPaths

class StereoBatchSampler(torch.utils.data.sampler.Sampler):
    """
    This sampler is designed to batch together all samples with a common (a) 2D molecular graph or (b) stereoisomer into the same batch.
    
    batch_size indicates either:
        (a) the number of molecular graphs per batch
        (b) the number of stereoisomers per batch

    The flag grouping = 'graphs' or 'stereoisomers' toggles the above options
    """
    
    def __init__(self, data_source, batch_size, grouping = 'stereoisomers'):
        self.data_source = deepcopy(data_source)
        self.data_source['batch_index'] = np.arange(0, len(self.data_source))
        self.batch_size = batch_size

        if grouping == 'graphs':
            self.groups = self.data_source.groupby(['SMILES_nostereo'])
        elif grouping == 'stereoisomers':
            self.groups = self.data_source.groupby(['ID'])
        else:
            raise Exception('grouping not defined')
        
    def __iter__(self): # returns list of lists of indices, with each inner list containing a batch of indices
        group_indices = [list(g[1].batch_index) for g in self.groups]
        np.random.shuffle(group_indices)

        batches = [list(chain(*group_indices[self.batch_size*i:self.batch_size*i+self.batch_size])) for i in range(math.ceil(len(self.groups)/self.batch_size))]
        return iter(batches)

    def __len__(self): # number of batches
        return math.ceil(len(self.groups) / self.batch_size) # includes last batch

class MaskedGraphDataset(torch_geometric.data.Dataset):
    def __init__(self, df, regression = '', stereoMask = True, mask_coordinates = False):
        super(MaskedGraphDataset, self).__init__()
        self.df = df
        self.stereoMask = stereoMask
        self.mask_coordinates = mask_coordinates
        self.regression = regression
        
    def get_all_paths(self, G, N = 3):
        # adapted from: https://stackoverflow.com/questions/28095646/finding-all-paths-walks-of-given-length-in-a-networkx-graph
        def findPaths(G,u,n):
            if n==0:
                return [[u]]
            paths = [[u]+path for neighbor in G.neighbors(u) for path in findPaths(G,neighbor,n-1) if u not in path]
            return paths
    
        allpaths = []
        for node in G:
            allpaths.extend(findPaths(G,node,N))
        return allpaths
    
    def process_mol(self, mol):
        # get internal coordinates for conformer, using all possible (forward) paths of length 2,3,4
        # Reverse paths (i.e., (1,2) and (2,1) or (1,2,3,4) and (4,3,2,1)) are not included when repeats == False
        # Note that we encode the reverse paths manually in alpha_encoder.py
        
        atom_symbols, edge_index, edge_features, node_features, bond_distances, bond_distance_index, bond_angles, bond_angle_index, dihedral_angles, dihedral_angle_index = embedConformerWithAllPaths(mol, repeats = False)
        
        bond_angles = bond_angles % (2*np.pi)
        dihedral_angles = dihedral_angles % (2*np.pi)
        
        data = torch_geometric.data.Data(x = torch.as_tensor(node_features), edge_index = torch.as_tensor(edge_index, dtype=torch.long), edge_attr = torch.as_tensor(edge_features))
        data.bond_distances = torch.as_tensor(bond_distances)
        data.bond_distance_index = torch.as_tensor(bond_distance_index, dtype=torch.long).T
        data.bond_angles = torch.as_tensor(bond_angles)
        data.bond_angle_index = torch.as_tensor(bond_angle_index, dtype=torch.long).T
        data.dihedral_angles = torch.as_tensor(dihedral_angles)
        data.dihedral_angle_index = torch.as_tensor(dihedral_angle_index, dtype=torch.long).T
        
        return data
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, key):
        mol = deepcopy(self.df.iloc[key].rdkit_mol_cistrans_stereo)
        
        data = self.process_mol(mol)
        
        if self.regression != '':
            #self.regression is the variable name of the supervised target in self.df
            y = torch.tensor(deepcopy(self.df.iloc[key][self.regression])) 

        if self.stereoMask:
            data.x[:, -9:] = 0.0
            data.edge_attr[:, -7:] = 0.0

        if self.mask_coordinates:
            data.bond_distances[:] = 0.0
            data.bond_angles[:] = 0.0
            data.dihedral_angles[:] = 0.0

        return (data, y) if self.regression != '' else data

class Dataset_3D_GNN(torch_geometric.data.Dataset):
    def __init__(self, df, regression = ''):
        super(Dataset_3D_GNN, self).__init__()
        self.df = df
        self.regression = regression
    
    def embed_mol(self, mol):
        if isinstance(mol, rdkit.Chem.rdchem.Conformer):
            mol = mol.GetOwningMol()
            conformer = mol
        elif isinstance(mol, rdkit.Chem.rdchem.Mol):
            mol = mol
            conformer = mol.GetConformer()

        # Edge Index
        adj = rdkit.Chem.GetAdjacencyMatrix(mol)
        adj = np.triu(np.array(adj, dtype = int)) #keeping just upper triangular entries from sym matrix
        array_adj = np.array(np.nonzero(adj), dtype = int) #indices of non-zero values in adj matrix
        edge_index = np.zeros((2, 2*array_adj.shape[1]), dtype = int) #placeholder for undirected edge list
        edge_index[:, ::2] = array_adj
        edge_index[:, 1::2] = np.flipud(array_adj)
        
        atoms = rdkit.Chem.rdchem.Mol.GetAtoms(mol)
        node_features = np.array([atom.GetAtomicNum() for atom in atoms]) # Z
        positions = np.array([conformer.GetAtomPosition(atom.GetIdx()) for atom in atoms]) # xyz positions
        
        return edge_index, node_features, positions # edge_index, Z, pos
        
    def process_mol(self, mol):
        edge_index, Z, pos = self.embed_mol(mol)
        data = torch_geometric.data.Data(x = torch.as_tensor(Z).unsqueeze(1), edge_index = torch.as_tensor(edge_index, dtype=torch.long))
        data.pos = torch.as_tensor(pos, dtype = torch.float)
        return data
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, key):
        mol = deepcopy(self.df.iloc[key].rdkit_mol_cistrans_stereo)
        data = self.process_mol(mol)
        
        if self.regression != '':
            #self.regression is the variable name of the supervised target in self.df
            y = torch.tensor(deepcopy(self.df.iloc[key][self.regression]))
        
        return (data, y) if self.regression != '' else data

class Sample_Map_To_Positives:
    def __init__(self, dataframe, isSorted=True, include_anchor = False): #isSorted vastly speeds up processing, but requires that the dataframe is sorted by SMILES_nostereo
        self.mapping = {}
        self.include_anchor = include_anchor
        
        for row_index, row in dataframe.iterrows():
            if isSorted:
                subset_df = dataframe.iloc[max(row_index-50, 0): row_index+50, :]
                
                if self.include_anchor == False:
                    positives = set(subset_df[(subset_df.ID == row.ID) & (subset_df.index.values != row_index)].index)
                else:
                    positives = set(subset_df[(subset_df.ID == row.ID)].index)
                
                self.mapping[row_index] = positives
                
    def sample(self, i, N=1, withoutReplacement=True): #sample positives
        if withoutReplacement:
            samples = random.sample(self.mapping[i], min(N, len(self.mapping[i])))
        else:
            samples = [random.choice(list(self.mapping[i])) for _ in range(N)]
        
        return samples

class Sample_Map_To_Negatives:
    def __init__(self, dataframe, isSorted=True): #isSorted vastly speeds up processing, but requires that the dataframe is sorted by SMILES_nostereo
        self.mapping = {}
        for row_index, row in dataframe.iterrows():
            if isSorted:
                negative_classes = []
                subset_df = dataframe.iloc[max(row_index-200, 0) : row_index+200, :]
                grouped_negatives = subset_df[(subset_df.SMILES_nostereo == row.SMILES_nostereo) & (subset_df.ID != row.ID)].groupby(by='ID', sort = False).groups.values()
                negative_classes = [set(list(group)) for group in grouped_negatives]
                self.mapping[row_index] = negative_classes
        
    def sample(self, i, N=1, withoutReplacement=True, stratified=True): #sample negatives
        if withoutReplacement:
            if stratified:
                samples = [random.sample(self.mapping[i][j], min(len(self.mapping[i][j]), N)) for j in range(len(self.mapping[i]))]
                samples = list(chain(*samples))
            else:
                population = list(chain(*[list(self.mapping[i][j]) for j in range(len(self.mapping[i]))]))
                samples = random.sample(population, min(len(population), N))
                
        else:
            if stratified:
                samples = [[random.choice(list(population)) for _ in range(N)] for population in self.mapping[i]]
                samples = list(chain(*samples))

            else:
                population = list(chain(*[list(self.mapping[i][j]) for j in range(len(self.mapping[i]))]))
                samples = [random.choice(population) for _ in range(N)]
            
        return samples

class SiameseBatchSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, data_source, batch_size, N_pos, N_neg, withoutReplacement = True, stratified = True):
        self.data_source = data_source
        self.positive_sampler = Sample_Map_To_Positives(data_source)
        self.negative_sampler = Sample_Map_To_Negatives(data_source)
        self.batch_size = batch_size
        self.withoutReplacement = withoutReplacement
        self.stratified = stratified
        self.N_pos = N_pos
        self.N_neg = N_neg
                
    def __iter__(self):
        groups = [[i, *self.positive_sampler.sample(i, N = self.N_pos, withoutReplacement = self.withoutReplacement), *self.negative_sampler.sample(i, N = self.N_neg, withoutReplacement = self.withoutReplacement, stratified = self.stratified)] for i in range(0, len(self.data_source))]
        np.random.shuffle(groups)
        batches = [list(chain(*groups[self.batch_size*i:self.batch_size*i+self.batch_size])) for i in range(math.floor(len(groups)/self.batch_size))]
        return iter(batches)

    def __len__(self): # number of batches
        return math.floor(len(self.data_source) / self.batch_size) #drops the last batch if it doesn't contain batch_size anchors

class NegativeBatchSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, data_source, batch_size, N_neg, withoutReplacement = True, stratified = True):
        self.data_source = data_source
        self.negative_sampler = Sample_Map_To_Negatives(data_source)
        self.batch_size = batch_size
        self.withoutReplacement = withoutReplacement
        self.stratified = stratified
        self.N_neg = N_neg
                
    def __iter__(self):
        groups = [[i, *self.negative_sampler.sample(i, N = self.N_neg, withoutReplacement = self.withoutReplacement, stratified = self.stratified)] for i in range(0, len(self.data_source))]
        np.random.shuffle(groups)
        batches = [list(chain(*groups[self.batch_size*i:self.batch_size*i+self.batch_size])) for i in range(math.floor(len(groups)/self.batch_size))]
        return iter(batches)

    def __len__(self): # number of batches
        return math.floor(len(self.data_source) / self.batch_size) #drops the last batch if it doesn't contain batch_size anchors


class SingleConformerBatchSampler(torch.utils.data.sampler.Sampler):
    # must be used with Sample_Map_To_Positives with include_anchor == True
    # Samples positives and negatives for each anchor, where the positives include the anchor
    
    # single_conformer_data_source is a dataframe consisting of just 1 conformer per stereoisomer
    # full_data_source is a dataframe consisting of all conformers for each stereoisomer
    # Importantly, single_conformer_data_source must be a subset of full_data_source, with the original indices
    
    def __init__(self, single_conformer_data_source, full_data_source, batch_size, N_pos = 0, N_neg = 1, withoutReplacement = True, stratified = True):
        self.single_conformer_data_source = single_conformer_data_source
        self.full_data_source = full_data_source
        
        self.positive_sampler = Sample_Map_To_Positives(full_data_source, include_anchor = True)
        self.negative_sampler = Sample_Map_To_Negatives(full_data_source)
        
        self.batch_size = batch_size
        self.withoutReplacement = withoutReplacement
        self.stratified = stratified
        self.N_pos = N_pos
        self.N_neg = N_neg
                
    def __iter__(self):
        groups = [[*self.positive_sampler.sample(i, N = 1 + self.N_pos, withoutReplacement = self.withoutReplacement), *self.negative_sampler.sample(i, N = self.N_neg, withoutReplacement = self.withoutReplacement, stratified = self.stratified)] for i in self.single_conformer_data_source.index.values]
        
        np.random.shuffle(groups)
        batches = [list(chain(*groups[self.batch_size*i:self.batch_size*i+self.batch_size])) for i in range(math.floor(len(groups)/self.batch_size))]
        return iter(batches)

    def __len__(self): # number of batches
        return math.floor(len(self.data_source) / self.batch_size) #drops the last batch if it doesn't contain batch_size anchors


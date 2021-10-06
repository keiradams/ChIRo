import torch
import torch.nn as nn
import torch_geometric
import numpy as np

string_to_object = {
    "torch.nn.LeakyReLU(negative_slope=0.01)": torch.nn.LeakyReLU(negative_slope=0.01),
    "torch.nn.LeakyReLU()": torch.nn.LeakyReLU(),
    "torch.nn.Identity()": torch.nn.Identity(),
    "torch.nn.ReLU()": torch.nn.ReLU(),
    "torch.nn.Sigmoid()": torch.nn.Sigmoid(),
    "torch.nn.Tanh()": torch.nn.Tanh()
}

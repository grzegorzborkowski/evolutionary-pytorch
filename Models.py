import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):

    def __init__(self, individual, number_of_layers, D_in, D_out, H):
        super(Model, self).__init__()
        self.number_of_layers = number_of_layers
        self.H = H
        self.individual = individual
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(D_in, H))
        for i in range(self.number_of_layers-1):
            self.layers.append(nn.Linear(H,H))
        self.layers.append(nn.Linear(H, D_out))

    def forward(self, x):
        y = x
        i=0
        for i in range(self.number_of_layers):
            if self.individual[i] == 0:
                y = F.sigmoid(self.layers[i](y))
            elif self.individual[i] == 1:
                y = F.relu(self.layers[i](y))
        y = F.softmax(self.layers[-1](y), dim=0)
        return y

class ModelFactory():

    def __init__(self, D_in, D_out, H):
        self.D_in = D_in
        self.D_out = D_out
        self.H = H

    def get_model(self, individual):
        return Model(individual, len(individual), self.D_in, self.D_out, self.H)

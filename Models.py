import torch.nn as nn
import torch.nn.functional as F

N, H = 64, 100

class Model(nn.Module):

    def __init__(self, individual, number_of_layers, D_in, D_out):
        super(Model, self).__init__()
        self.number_of_layers = number_of_layers
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
                y = F.tanh(self.layers[i](y))
            else:
                y = F.relu(self.layers[i](y))
        y = F.softmax(self.layers[-1](y), dim=0)
        return y


class ModelFactory():

    def __init__(self, D_in, D_out):
        self.D_in = D_in
        self.D_out = D_out

    def get_model(self, individual):
        ones = 0
        for elem in individual:
            if elem == 1:
                ones += 1
        ones = max(ones, 1) # we can't have all zeros
        return Model(individual, ones, self.D_in, self.D_out)

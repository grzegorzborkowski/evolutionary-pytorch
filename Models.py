import torch.nn as nn
import torch.nn.functional as F

N, D_in, H, D_out = 64, 4, 100, 3

class Model(nn.Module):

    def __init__(self, number_of_layers):
        super(Model, self).__init__()
        self.number_of_layers = number_of_layers

    def forward(self, x):
        x = F.relu((nn.Linear(D_in, H)(x)))

        for i in range(self.number_of_layers):
            x = F.relu(nn.Linear(H, H)(x))

        x = F.softmax((nn.Linear(H, D_out)(x)))
        return x


class ModelFactory():

    def get_model(self, individual):
        ones = 0
        for elem in individual:
            if elem == 1:
                ones += 1
        return Model(ones)

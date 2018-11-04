from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import train_test_split
import torch
from abc import ABC


class AbstractScikitDataProcessor(ABC):

    def __init__(self, data=None):
        self.data = data
        if data is not None:
            train_X, test_X, train_y, test_y = train_test_split(self.data.data, data.target, test_size=0.8, random_state=5)
            self.train_X = torch.Tensor(train_X)
            self.test_X = torch.Tensor(test_X)
            self.train_y = torch.Tensor(train_y).long()
            self.test_y = torch.Tensor(test_y).long()

    def get_data(self):
        return (self.train_X, self.test_X, self.train_y, self.test_y)


class Iris(AbstractScikitDataProcessor):

    def __init__(self):
        super().__init__(load_iris())


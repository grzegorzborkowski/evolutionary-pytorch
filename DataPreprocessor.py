from sklearn.datasets import load_iris, load_wine, load_digits
from sklearn.model_selection import train_test_split
import torch
from abc import ABC

class AbstractScikitDataProcessor(ABC):

    def __init__(self, data=None):
        self.data = data
        if data is not None:
            train_X, test_X, train_y, test_y = train_test_split(self.data.data, data.target, test_size=0.8, random_state=5)
            
            self.train_y_raw = train_y
            self.train_X = torch.Tensor(train_X)
            self.test_X = torch.Tensor(test_X)
            self.train_y = torch.Tensor(train_y).long()
            self.test_y = torch.Tensor(test_y).long()

    def get_data(self):
        return (self.train_X, self.test_X, self.train_y, self.test_y)

    def get_input_layer_size(self):
       return self.train_X.shape[1]

    def get_output_layer_size(self):
        return len(list(set(self.train_y_raw)))

class Iris(AbstractScikitDataProcessor):

    def __init__(self):
        super().__init__(load_iris())

class Wine(AbstractScikitDataProcessor):

    def __init__(self):
        super().__init__(load_wine())

class Mnist(AbstractScikitDataProcessor):

    def __init__(self):
        super().__init__(load_digits())
import argparse

class HyperParams:

    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--mutation_probability', type=float, default=0.25)
        self.parser.add_argument('--crosover_probability', type=float, default=0.8)
        self.parser.add_argument('--population_size', type=int, default=10)
        self.parser.add_argument('--tournament_size', type=int, default=3)
        self.parser.add_argument('--number_of_generations', type=int, default=10)
        self.parser.add_argument('--max_number_of_layers', type=int, default=10)
        self.parser.add_argument('--learning_rate', type=float, default=0.001)
        self.parser.add_argument('--hidden_units', type=int, default=100)
        self.parser.add_argument('--number_of_epochs', type=int, default=250)

    def parse_args(self):
        self.args = self.parser.parse_args()
        args_params = vars(self.args)
        self.mutation_probability = args_params['mutation_probability']
        self.crosover_probability = args_params['crosover_probability']
        self.population_size = args_params['population_size']
        self.tournament_size = args_params['tournament_size']
        self.number_of_generations = args_params['number_of_generations']
        self.max_number_of_layers = args_params['max_number_of_layers']
        self.learning_rate = args_params['learning_rate']
        self.hidden_units = args_params['hidden_units']
        self.number_of_epochs = args_params['number_of_epochs']


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
        self.parser.add_argument('--file_path', type=str, default="results.csv")

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
        self.file_path = args_params['file_path']

    def params_str(self):
        return str(self.mutation_probability) + "|" + str(self.crosover_probability) + "|" + str(self.population_size) + "|" + str(self.tournament_size) + "|" + str(self.number_of_generations) + "|" + str(self.max_number_of_layers) + "|" + str(self.learning_rate) + "|" + str(self.hidden_units) + "|" + str(self.number_of_epochs) + "|"

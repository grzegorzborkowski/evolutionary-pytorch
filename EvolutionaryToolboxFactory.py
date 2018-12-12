from deap import base, creator, tools
import numpy

class EvolutionaryToolboxFactory():

    def get_toolbox(self, mutation_probability, tournament_size, max_number_of_layers):
        self.max_number_of_layers = max_number_of_layers
        creator.create("FitnessMax", base.Fitness, weights=(1.0,)) # maximizing
        creator.create("Individual", list, fitness=creator.FitnessMax)

        toolbox = base.Toolbox()

        toolbox.register("individual", self.__initInd__, icls=creator.Individual)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        toolbox.register("mate", tools.cxTwoPoint)

        # register a mutation operator with a probability to
        # flip each attribute/gene of 0.25
        toolbox.register("mutate", tools.mutFlipBit, indpb=mutation_probability)

        # operator for selecting individuals for breeding the next
        # generation: each individual of the current generation
        # is replaced by the 'fittest' (best) of three individuals
        # drawn randomly from the current generation.
        toolbox.register("select", tools.selTournament, tournsize=tournament_size)

        return toolbox

    def __initInd__(self, icls):
        how_many = numpy.random.random_integers(2, self.max_number_of_layers)
        list = []
        for i in range(how_many):
            list.append(numpy.random.random_integers(0, 1))
        return icls(list)


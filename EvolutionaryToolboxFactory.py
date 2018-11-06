from deap import base
from deap import creator
from deap import tools
import random
import numpy

class EvolutionaryToolboxFactory():

    def get_toolbox(self):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,)) # maximizing
        creator.create("Individual", list, fitness=creator.FitnessMax)

        toolbox = base.Toolbox()

        toolbox.register("individual", self.__initInd__, icls=creator.Individual)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        toolbox.register("mate", tools.cxTwoPoint)

        # register a mutation operator with a probability to
        # flip each attribute/gene of 0.05
        toolbox.register("mutate", tools.mutFlipBit, indpb=0.25)

        # operator for selecting individuals for breeding the next
        # generation: each individual of the current generation
        # is replaced by the 'fittest' (best) of three individuals
        # drawn randomly from the current generation.
        toolbox.register("select", tools.selTournament, tournsize=3)

        return toolbox

    def __initInd__(self, icls):
        how_many = numpy.random.random_integers(2, 10)
        list = []
        for i in range(how_many):
            list.append(numpy.random.random_integers(0, 1))
        return icls(list)


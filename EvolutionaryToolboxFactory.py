from deap import base
from deap import creator
from deap import tools
import random


class EvolutionaryToolboxFactory():

    def get_toolbox(self):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        toolbox = base.Toolbox()
        toolbox.register("attr_bool", random.randint, 0, 1)
        toolbox.register("individual", tools.initRepeat, creator.Individual,
                         toolbox.attr_bool, random.randint(10, 30))
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        toolbox.register("mate", tools.cxTwoPoint)

        # register a mutation operator with a probability to
        # flip each attribute/gene of 0.05
        toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)

        # operator for selecting individuals for breeding the next
        # generation: each individual of the current generation
        # is replaced by the 'fittest' (best) of three individuals
        # drawn randomly from the current generation.
        toolbox.register("select", tools.selTournament, tournsize=3)

        return toolbox

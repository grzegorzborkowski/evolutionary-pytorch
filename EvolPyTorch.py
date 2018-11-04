import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from deap import base, creator, tools, algorithms
import random
import numpy
import Models
import DataPreprocessor
import EvolutionaryToolboxFactory

import random

torch.set_default_tensor_type('torch.FloatTensor')

#torch.manual_seed(7)

dataPreprocessor = DataPreprocessor.Wine()
train_X, test_X, train_y, test_y = dataPreprocessor.get_data()
input_layer_size = dataPreprocessor.get_input_layer_size()
output_layer_size = dataPreprocessor.get_output_layer_size()

# the goal ('fitness') function to be maximized
def evalOneMax(individual):
    print ("Evaluatio individual", individual)
    model_factory = Models.ModelFactory(input_layer_size, output_layer_size)
    model = model_factory.get_model(individual)
    loss_fn = nn.CrossEntropyLoss()

    learning_rate = 0.1

    for t in range(250):
        y_pred = model(train_X)
        loss = loss_fn(y_pred, train_y)

        if t % 25 == 0:
            pass
            #print (t, loss.item())

        model.zero_grad()
        loss.backward()

        with torch.no_grad():
            for param in model.parameters():
                param.data -= learning_rate * param.grad

    predict_out = model(test_X)
    _, predict_y = torch.max(predict_out, 1)
    accuracy = accuracy_score(test_y, predict_y)
    print (accuracy)
    return accuracy,


def main():
    toolbox_factory = EvolutionaryToolboxFactory.EvolutionaryToolboxFactory()
    toolbox = toolbox_factory.get_toolbox()
    toolbox.register("evaluate", evalOneMax)

    pop = toolbox.population(n=10)

    CXPB, MUTPB = 0.5, 0.2

    print("Start of evolution")

    # Evaluate the entire population
    fitnesses = list(map(toolbox.evaluate, pop))
    print ("fitness", fitnesses)
    for ind, fit in zip(pop, fitnesses):
        print ("ind", "fit", ind, fit)
        ind.fitness.values = fit

    print("  Evaluated %i individuals" % len(pop))

    # Extracting all the fitnesses of 
    fits = [ind.fitness.values[0] for ind in pop]

    # Variable keeping track of the number of generations
    g = 0

    # Begin the evolution
    while max(fits) < 5 and g < 5:
        # A new generation
        g = g + 1
        print("-- Generation %i --" % g)

        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))

        offspring = algorithms.varAnd(pop, toolbox, CXPB, MUTPB)
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            #print ("ind", "fit", ind, fit)
            ind.fitness.values = fit

        print("  Evaluated %i individuals" % len(invalid_ind))

        # The population is entirely replaced by the offspring
        pop[:] = offspring

        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in pop]

        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x*x for x in fits)
        std = abs(sum2 / length - mean**2)**0.5

        print("  Min %s" % min(fits))
        print("  Max %s" % max(fits))
        print("  Avg %s" % mean)
        print("  Std %s" % std)

    print("-- End of (successful) evolution --")

    best_ind = tools.selBest(pop, 1)[0]
    print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))

if __name__ == "__main__":
    main()



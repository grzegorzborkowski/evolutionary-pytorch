import torch
from deap import base, creator, tools, algorithms
import Models
import DataPreprocessor
import EvolutionaryToolboxFactory
import ModelEvaluator

torch.set_default_tensor_type('torch.FloatTensor')

def main():
    dataPreprocessor = DataPreprocessor.Wine()
    train_X, test_X, train_y, test_y = dataPreprocessor.get_data()
    input_layer_size = dataPreprocessor.get_input_layer_size()
    output_layer_size = dataPreprocessor.get_output_layer_size()

    model_evaluator = ModelEvaluator.ModelEvaluator()

    toolbox_factory = EvolutionaryToolboxFactory.EvolutionaryToolboxFactory()
    toolbox = toolbox_factory.get_toolbox()
    toolbox.register("evaluate", model_evaluator.evalOneMax)

    pop = toolbox.population(n=10)

    CXPB, MUTPB = 0.8, 0.5

    print("Start of evolution")

    # Evaluate the entire population
    fitnesses = []
    for el in pop:
        print (el)
        result = toolbox.evaluate(el, input_layer_size, output_layer_size, train_X, test_X, train_y, test_y)
        fitnesses.append(result)

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

        fitnesses = []
        for el in invalid_ind:
            result = toolbox.evaluate(el, input_layer_size, output_layer_size, train_X, test_X, train_y, test_y)
            fitnesses.append(result)

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



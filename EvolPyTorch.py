import torch
from deap import base, creator, tools, algorithms
import Models
import DataPreprocessor
import EvolutionaryToolboxFactory
import ModelEvaluator
import HyperParams

torch.set_default_tensor_type('torch.FloatTensor')

DEBUG = False

def main():
    hyper_params = HyperParams.HyperParams()
    hyper_params.parse_args()
    dataPreprocessor = DataPreprocessor.Wine()
    train_X, test_X, train_y, test_y = dataPreprocessor.get_data()
    input_layer_size = dataPreprocessor.get_input_layer_size()
    output_layer_size = dataPreprocessor.get_output_layer_size()

    model_evaluator = ModelEvaluator.ModelEvaluator()

    toolbox_factory = EvolutionaryToolboxFactory.EvolutionaryToolboxFactory()
    toolbox = toolbox_factory.get_toolbox(hyper_params.mutation_probability, hyper_params.tournament_size, hyper_params.max_number_of_layers)
    toolbox.register("evaluate", model_evaluator.evalOneMax)

    pop = toolbox.population(n=hyper_params.population_size)

    CXPB, MUTPB = hyper_params.crosover_probability, hyper_params.mutation_probability
    
    if DEBUG:
        print("Start of evolution")

    # Evaluate the entire population
    fitnesses = []
    for el in pop:
        if DEBUG:
            print (el)
        result = toolbox.evaluate(el, hyper_params.number_of_epochs, 
            hyper_params.learning_rate, input_layer_size, output_layer_size, hyper_params.hidden_units, train_X, test_X, train_y, test_y)
        fitnesses.append(result)
    if DEBUG:
        print ("fitness", fitnesses)
    for ind, fit in zip(pop, fitnesses):
        if DEBUG:
            print ("ind", "fit", ind, fit)
        ind.fitness.values = fit
    if DEBUG:
        print("  Evaluated %i individuals" % len(pop))

    # Extracting all the fitnesses of 
    fits = [ind.fitness.values[0] for ind in pop]

    # Variable keeping track of the number of generations
    g = 0

    # Begin the evolution
    while g < hyper_params.number_of_generations:
        # A new generation
        g = g + 1
        if DEBUG:
            print("-- Generation %i --" % g)

        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))

        offspring = algorithms.varAnd(pop, toolbox, CXPB, MUTPB)
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]

        fitnesses = []
        for el in invalid_ind:
            result = toolbox.evaluate(el, hyper_params.number_of_epochs,
            hyper_params.learning_rate, input_layer_size, output_layer_size, hyper_params.hidden_units, train_X, test_X, train_y, test_y)
            fitnesses.append(result)

        for ind, fit in zip(invalid_ind, fitnesses):
            #print ("ind", "fit", ind, fit)
            ind.fitness.values = fit
        if DEBUG:
            print("  Evaluated %i individuals" % len(invalid_ind))

        # The population is entirely replaced by the offspring
        pop[:] = offspring

        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in pop]

        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x*x for x in fits)
        std = abs(sum2 / length - mean**2)**0.5
        if DEBUG:
            print("  Min %s" % min(fits))
            print("  Max %s" % max(fits))
            print("  Avg %s" % mean)
            print("  Std %s" % std)
    if DEBUG:
        print("-- End of (successful) evolution --")

    best_ind = tools.selBest(pop, 1)[0]
    if DEBUG:
        print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))
    else:
        with open(hyper_params.file_path, 'a') as file:
            file.write(hyper_params.params_str())
            file.write(str(best_ind) + "," + str(best_ind.fitness.values)[1:][:-2] + "\n")

if __name__ == "__main__":
    main()



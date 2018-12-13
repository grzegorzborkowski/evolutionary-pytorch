for mutation in 0.2 0.3 0.4 0.5
    do
        for crossover in 0.5 
        do
            for population_size in 10 25 50
            do
                for tournament in 1 3 5 7
                do
                    for generation in 10
                    do
                        for layer in 5 10
                        do 
                            for learning_rate in 0.000001
                            do
                                for hidden_units in 32 64 100
                                do 
                                    for epochs in 50
                                    do
                                        for exectutions in {1..5..1}
                                        do
                                            python3 EvolPyTorch.py --mutation_probability $mutation --crosover_probability $crossover --population_size $population_size --tournament_size $tournament --number_of_generations $generation --max_number_of_layers $layer --learning_rate $learning_rate --hidden_units $hidden_units --number_of_epochs $epochs 
                                        done
                                    done

                                done

                            done
                        done

                    done
                done
            done

        done
done
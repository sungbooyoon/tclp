import numpy as np
import functions
import random


def cal_pop_fitness(pop):
    # Calculating the fitness value of each solution in the current population.
    # The fitness function calculates the sum of products between each input and its corresponding weight.
    scores = np.zeros((pop.shape[0], 2))
    for i, v in enumerate(pop):
        crane_model = functions.get_crane(v)
        fee = functions.cal_total_fee(v)
        time_list = functions.get_total_operation_time(v)
        time = functions.get_max_operation_time(time_list)
        scores[i, 0] = fee
        scores[i, 1] = time
    return scores


def select_mating_pool(pop, pareto, min_number_parents):
    # Selecting the best individuals in the current generation as parents for producing the offspring of the next generation.
    if pareto.shape[0] >= min_number_parents:
        parents = pareto
        return parents.astype(np.int64)
    else:
        parents_list = pareto.tolist()
        while len(parents_list) < min_number_parents:
            parents_list = functions.make_pareto_parents(parents_list, pop, min_number_parents)
            if len(parents_list) >= min_number_parents:
                break
        return np.array(parents_list)


def crossover(parents, offspring_size):
    offspring = np.zeros(offspring_size)
    for k in range(offspring_size[0]):
        # Index of the first parent to mate.
        parent1_idx = k % parents.shape[0]
        # Index of the second parent to mate.
        parent2_idx = (k + 1) % parents.shape[0]
        crane_location_idx1 = np.where(parents[parent1_idx, :] != 0)[0]
        crane_location_idx2 = np.where(parents[parent2_idx, :] != 0)[0]
        crane_location_idx = np.concatenate((crane_location_idx1, crane_location_idx2), axis=0)
        crane_location_idx = list(set(list(crane_location_idx)))
        crossover_idx = random.sample(crane_location_idx, len(list(crane_location_idx1)))
        for i in crossover_idx:
            if i in crane_location_idx1:
                offspring[k, i] = parents[parent1_idx, i]
            else:
                offspring[k, i] = parents[parent2_idx, i]
    return offspring.astype(np.int64)


def mutation(offspring_crossover, crane_data, mutation_rate):
    # Mutation changes a number of genes as defined by the num_mutations argument. The changes are random.
    for idx in range(offspring_crossover.shape[0]):
        crane_location_idx = np.where(offspring_crossover[idx, :] != 0)[0]
        location_list = []
        for i in range(offspring_crossover.shape[1]):
            location_list.append(i)
        for i in list(crane_location_idx):
            location_list.remove(i)
        rnd = np.random.rand(len(location_list))
        mutate_at = rnd < mutation_rate
        idx_list = np.where(mutate_at == True)[0]
        if len(idx_list) != 0:
            index = random.choice(idx_list)
            x = location_list[index]
            y = random.choice(crane_location_idx)
            offspring_crossover[idx, x] = random.randrange(1, len(crane_data)+1)
            offspring_crossover[idx, y] = 0
        else:
            pass

    return offspring_crossover

import numpy as np
import random
import sys
from itertools import combinations
import json
import math
from tqdm import tqdm
import time


class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        with open(file_path, "r") as json_file:
            if json_file is None:
                print(f"***Error: No json file located in {file_path}.")
                sys.exit()
            self.all_data = json.load(json_file)

    def get_crane_location(self):
        _crane_location = self.all_data["tc_location_list"]
        self.crane_location = np.array(
            [[crane["x"], crane["y"]] for crane in _crane_location]
        )
        print("Crane location shape: ", self.crane_location.shape)
        return self.crane_location

    def get_building_info(self):
        self.building_info = self.all_data["building_list"]
        print("Number of buildings: ", len(self.building_info))
        return self.building_info

    def get_unit_location(self):
        _unit_location = self.all_data["unit_location_list"]
        self.unit_location = []
        for buildings in _unit_location:
            self.unit_location.append(
                np.array([[building["x"], building["y"]] for building in buildings])
            )
        print(
            "Unit location shape: ",
            [building.shape for building in self.unit_location],
        )
        return self.unit_location

    def get_crane_model(self):
        crane_model = self.all_data["tc_list"]

        # Replace string values in dict to int or float
        for model in crane_model:
            model["idx"] = int(model["idx"])
            model["ton"] = float(model["ton"])
            model["height"] = float(model["height"])
            model["vr"] = float(model["vr"])
            model["vw"] = float(model["vw"])
            model["vv"] = float(model["vv"])
            model["radius"] = float(model["radius"])
            model["rmax"] = float(model["rmax"])
            model["rentalcost"] = float(model["rentalcost"])
            model["fixedcost"] = float(model["fixedcost"])

        print("Number of cranes in crane DB: ", len(crane_model))
        return crane_model

    def get_parameters(self):
        parameters = self.all_data["simulation_param"]
        print("Number of parameters: ", len(parameters))
        return parameters

    def __len__(self):
        return len(self.all_data)


class Crane:
    def __init__(
        self, crane_location, unit_location, crane_model, building_info, parameters
    ):
        self.crane_location = crane_location
        self.unit_location = unit_location
        self.building_info = building_info
        self.num_building = len(building_info)
        self.building_floor = [info["num_floor"] for info in self.building_info]
        self.crane_model = crane_model
        self.num_crane = parameters["num_crane"]
        self.min_crane_distance = parameters["min_crane_distance"]
        self.num_modules = parameters["num_modules"]
        self.max_cost = parameters["max_cost"]
        self.max_duration = parameters["max_duration"]
        self.lifting_time = parameters["lifting_time"]
        self.installing_time = parameters["installing_time"]
        self.finishing_time = parameters["finishing_time"]

    def set_layout(self, layout):
        """
        cranes_idx : 1 ~ 18
        layout : [(Model, Location)] form. Type = list
        """
        self.layout = layout
        # print("Layout updated!")
        # print("Layout: ", self.layout)

    def random_layout(self):
        models_idx = np.random.randint(
            1, len(self.crane_model) + 1, self.num_crane
        ).tolist()
        locations_idx = np.random.choice(
            len(self.crane_location), self.num_crane, replace=False
        ).tolist()
        random_layout = list(zip(models_idx, locations_idx))
        return random_layout

    def manual_layout(self, modes_idx, locations_idx):
        manual_layout = list(zip(models_idx, locations_idx))
        return manual_layout

    def chromosome_layout(self, chromosome):
        locations_idx = np.where(chromosome != 0)[0].tolist()
        models_idx = [chromosome[location_idx] for location_idx in locations_idx]
        chromosome_layout = list(zip(models_idx, locations_idx))
        return chromosome_layout

    def get_chromosome(self):
        chromosome = np.zeros((len(self.crane_location),), dtype=np.uint8)
        for model_idx, location_idx in self.layout:
            if model_idx == 0:
                print("***Error: 0 is in the index of the cranes.")
                sys.exit()
            chromosome[location_idx] = model_idx
        return chromosome

    def get_locations(self):
        locations = []
        for model_idx, location_idx in self.layout:
            if model_idx == 0:
                print("***Error: 0 is in the index of the cranes.")
                sys.exit()
            locations.append(self.crane_location[location_idx])
        return np.array(locations)

    def get_models(self):
        models = []
        for model_idx, location_idx in self.layout:
            if model_idx == 0:
                print("***Error: 0 is in the index of the cranes.")
                sys.exit()
            models.append(self.crane_model[model_idx - 1])
        return np.array(models)

    def _cal_distance(self, location1, location2, array=False):
        if array:
            distance_x = location1[:, :, 0] - location2[:, :, 0]
            distance_y = location1[:, :, 1] - location2[:, :, 2]
            distance = np.sqrt(np.square(distance_x) + np.square(distance_y))
            return distance
        distance_x = location1[0] - location2[0]
        distance_y = location1[1] - location2[1]
        distance = math.sqrt((distance_x ** 2 + distance_y ** 2))
        return distance

    def _get_feasible(self, unit_location, locations, models):
        distances = [
            self._cal_distance(unit_location, crane_location)
            for crane_location in locations
        ]
        feasible = [distances[i] <= models[i]["rmax"] for i in range(len(distances))]
        return feasible

    def allocation(self):
        locations = self.get_locations()
        models = self.get_models()
        task_allocation_1st_floor = []
        model_allocation_1st_floor = []
        crane_idx_allocation_1st_floor = []

        for buildings in self.unit_location:
            task_allocation_single = np.zeros(self.num_crane, dtype=np.int32)
            model_allocation_single = []
            crane_idx_allocation_single = []

            for unit_location in buildings:
                feasible = self._get_feasible(unit_location, locations, models)
                if any(feasible):
                    feasible_crane_idx = [
                        i for i, v in enumerate(feasible) if v is True
                    ]  # [0, 1]
                    _task_allocation_single = [
                        task_allocation_single[crane_idx]
                        for crane_idx in feasible_crane_idx
                    ]  # [14, 12]
                    _model_allocation_single = feasible_crane_idx[
                        _task_allocation_single.index(min(_task_allocation_single))
                    ]  # 1
                    model_allocation_single.append(
                        models[_model_allocation_single]["idx"]
                    )
                    crane_idx_allocation_single.append(_model_allocation_single)
                    task_allocation_single[_model_allocation_single] += 1
                    continue

            model_allocation_1st_floor.append(np.array(model_allocation_single))
            crane_idx_allocation_1st_floor.append(np.array(crane_idx_allocation_single))
            task_allocation_1st_floor.append(task_allocation_single)
        task_allocation_1st_floor = np.array(task_allocation_1st_floor)

        if self.num_building != len(
            task_allocation_1st_floor
        ) or self.num_building != len(model_allocation_1st_floor):
            print("***Error: Allocation is done incorrectly.")
            sys.exit()

        task_allocation = []
        model_allocation = []
        crane_idx_allocation = []

        for building_idx in range(self.num_building):
            _task_allocation = (
                task_allocation_1st_floor[building_idx]
                * self.building_floor[building_idx]
            ).tolist()
            models_idx = [model["idx"] for model in models]
            task_allocation.append(list(zip(models_idx, _task_allocation)))

            _model_allocation = [
                model_allocation_1st_floor[building_idx]
                for _ in range(self.building_floor[building_idx])
            ]
            model_allocation.append(np.vstack(_model_allocation))
            _crane_idx_allocation = [
                crane_idx_allocation_1st_floor[building_idx]
                for _ in range(self.building_floor[building_idx])
            ]
            crane_idx_allocation.append(np.vstack(_crane_idx_allocation))
        task_allocation = np.array(task_allocation)
        return task_allocation, model_allocation, crane_idx_allocation

    def check_unit_distance(self):
        locations = self.get_locations()
        models = self.get_models()
        for building in self.unit_location:
            for unit_location in building:
                feasible = self._get_feasible(unit_location, locations, models)
                # print(unit_location)
                # print(locations.ravel())
                # print([models[i]['rmax']for i in range(len(models))])
                # print(feasible)
                if any(feasible):
                    continue
                else:

                    # print("unit distance")
                    return False
        return True

    def check_crane_distance(self):
        locations = self.get_locations()
        crane_combinations = list(combinations(locations.tolist(), 2))
        for crane_combination in crane_combinations:
            if (
                self._cal_distance(crane_combination[0], crane_combination[1])
                < self.min_crane_distance
            ):
                # print("crane distance")
                return False
            continue
        return True

    def check_allocation(self):
        task_allocation, _, _ = self.allocation()
        task_sum = np.sum(task_allocation, axis=0)[:, 1]
        is_task_none = task_sum == 0
        if any(is_task_none):
            # print("allocation")
            return False
        return True

    def check_height(self):
        models = self.get_models()
        building_height = np.max(np.array(self.building_floor)) * 0.3
        building_higher = [model["height"] < building_height for model in models]
        if any(building_higher):
            # print("height")
            return False
        return True

    def check_layout(self):
        if (
            self.check_unit_distance()
            and self.check_crane_distance()
            and self.check_allocation()
            and self.check_height()
        ):
            return True
        return False

    def _get_unit_xyz(self):
        unit_xyz = []
        for building_idx, floors in enumerate(self.building_floor):
            _unit_xy = [
                [self.unit_location[building_idx]]
                for _ in range(self.building_floor[building_idx])
            ]
            unit_xy = np.vstack(_unit_xy)
            _unit_z = np.arange(0, 3 * floors, 3).reshape(-1, 1)
            _unit_z = [_unit_z for _ in range(len(self.unit_location[building_idx]))]
            unit_z = np.hstack(_unit_z).reshape(floors, -1, 1)
            unit_xyz.append(np.append(unit_xy, unit_z, axis=2))
        return unit_xyz

    def _model_info_to_array(self, model_allocation):
        vv = model_allocation.copy().astype(np.float32)
        vr = model_allocation.copy().astype(np.float32)
        vw = model_allocation.copy().astype(np.float32)
        rentalcost = model_allocation.copy()
        fixedcost = model_allocation.copy()

        for model in self.crane_model:
            vv[np.where(vv == model["idx"])] = model["vv"]
            vr[np.where(vr == model["idx"])] = model["vr"]
            vw[np.where(vw == model["idx"])] = model["vw"]
            rentalcost[np.where(rentalcost == model["idx"])] = model["rentalcost"]
            fixedcost[np.where(fixedcost == model["idx"])] = model["fixedcost"]
        return vv, vr, vw, rentalcost, fixedcost

    def cal_function(self):
        models = self.get_models()
        locations = self.get_locations()
        supply_height = 0
        alpha = 0.25  # nadoushani
        beta = 1.0

        unit_xyz = self._get_unit_xyz()
        _, model_allocation, crane_idx_allocation = self.allocation()

        rental_cost = []
        time = np.zeros(self.num_crane, dtype=np.float64)

        for building_idx in range(self.num_building):
            _model_allocation = model_allocation[building_idx]
            _crane_idx_allocation = crane_idx_allocation[building_idx]
            location = locations[_crane_idx_allocation]
            _unit_xyz = unit_xyz[building_idx]
            vv, vr, vw, rentalcost, fixedcost = self._model_info_to_array(
                _model_allocation
            )
            tl = self.lifting_time
            tu = self.installing_time
            tf = self.finishing_time
            tv = abs(_unit_xyz[:, :, 2] - supply_height) / vv
            pd = self._cal_distance(location, _unit_xyz, array=True)
            ps = np.ones((location.shape[0], location.shape[1]), dtype=np.float64) * 3
            lcr = random.uniform(
                abs(pd - ps) + 0.01, pd + ps - 0.01
            )  # subject to change
            tr = abs(pd - ps) / vr
            x = (-(lcr ** 2) + pd ** 2 + ps ** 2) / (2 * pd * ps)
            tw = 1 / (vw * 10) * np.arccos(x)
            th = np.maximum(tr, tw) + alpha * np.minimum(tr, tw)
            t = (np.maximum(th, tv) + beta * np.minimum(th, tv) + tl + tu) + tf
            time[_crane_idx_allocation] += t / 60
            rental_cost.append(np.sum(t * rentalcost * 1000 / 8 / 60))

        fixed_cost = [model["fixedcost"] * 1000 for model in models]
        total_cost = sum(rental_cost) + sum(fixed_cost)
        total_cost = total_cost * 10  # KRW
        max_time = max(time)  # days? originally hours, not divided by 10
        return total_cost / 1000000, max_time * 10

    def collision(self):
        models = self.get_models().tolist()
        locations = self.get_locations().tolist()
        shared_space = 0
        crane_combinations = list(combinations(locations, 2))
        for combination in crane_combinations:
            distance = self._cal_distance(combination[0], combination[1])
            r1 = models[locations.index(combination[0])]["rmax"]
            r2 = models[locations.index(combination[1])]["rmax"]
            shared_space += self._cal_shared_space(distance, r1, r2)
        return shared_space

    def cal_shared_space(self, distance, r1, r2):
        if distance < r1 + r2:
            a = r1 ** 2
            b = r2 ** 2
            x = (a - b + distance ** 2) / (2 * distance)
            z = x ** 2
            y = math.sqrt(abs(a - z))
            if distance <= abs(r1 - r2):
                return math.pi * min(a, b)
            shared_space = (
                10 * a * math.asin(y / r1)
                + b * math.asin(y / r2)
                - y * (x + math.sqrt(abs(z + b - a)))
            )
            return shared_space
        else:
            return 0


class GeneticAlgorithms(Crane):
    def __init__(
        self, crane_location, unit_location, crane_model, building_info, parameters
    ):
        super().__init__(
            crane_location, unit_location, crane_model, building_info, parameters
        )
        self.num_population = parameters["num_population"]
        self.min_parents = parameters["min_parents"]
        self.generation = parameters["generation"]
        self.mutation_rate = parameters["mutation_rate"]

    def set_population(self, population):
        if len(population) != self.num_population:
            print(
                "***Error: The number of given population is different from the default number of population."
            )
            sys.exit()
        self.population = population

    def fitness(self, pool=None):
        if pool is None:
            population = self.population
        else:
            population = pool
        scores = np.zeros((len(population), 2))
        for idx, chromosome in enumerate(population):
            chromosome_layout = self.chromosome_layout(chromosome)
            self.set_layout(chromosome_layout)
            cost, time = self.cal_function()
            scores[idx] = cost, time
        return scores

    def selection(self, pareto):
        start_selection = time.time()
        if len(pareto) >= self.min_parents:
            parents = pareto
        else:
            while 1:
                check = time.time()
                parents = self.get_pareto_parents(pareto)
                if check - start_selection > 5:
                    print("***Error: Selection error.")
                    break
                if len(parents) >= self.min_parents:
                    break
        return parents

    def crossover(self, parents, num_children):
        children = np.zeros((num_children, len(self.crane_location)))
        for child_idx in range(num_children):
            parent1_idx = child_idx % len(parents)
            # Index of the second parent to mate.
            parent2_idx = (child_idx + 1) % len(parents)
            location_idx1 = np.where(parents[parent1_idx, :] != 0)[0]
            location_idx2 = np.where(parents[parent2_idx, :] != 0)[0]
            location_idx = np.hstack((location_idx1, location_idx2))
            location_idx = np.array(list(set(list(location_idx))))
            crossover_idx = np.random.choice(
                location_idx, len(location_idx1), replace=False
            )
            for idx in crossover_idx:
                if idx in location_idx1:
                    children[child_idx, idx] = parents[parent1_idx, idx]
                else:
                    children[child_idx, idx] = parents[parent2_idx, idx]
        children_crossover = children.astype(np.int32)
        return children_crossover

    # def mutation(self):
    def mutation(self, children_crossover):
        for child_idx, children in enumerate(children_crossover):
            zeros_idx = np.where(children == 0)[0]
            old_idx = np.where(children != 0)[0]
            rnd = np.random.rand(len(zeros_idx.tolist()))
            mutate_at = rnd < self.mutation_rate
            mutation_idx = np.where(mutate_at == True)[0]
            if len(mutation_idx) != 0:
                new_idx = np.random.choice(mutation_idx, 1)
                to_new_idx = zeros_idx[new_idx]
                to_zero_idx = np.random.choice(old_idx, 1)
                children_crossover[child_idx, to_new_idx] = random.randrange(
                    1, len(self.crane_model) + 1
                )
                children_crossover[child_idx, to_zero_idx] = 0
            else:
                pass
        children_mutation = children_crossover
        return children_mutation

    def identify_pareto(self, scores, pool=None):
        if pool is None:
            population = self.population
        else:
            population = pool
        pareto_front = np.ones(len(population), dtype=bool)
        for i in range(len(population)):
            # Loop through all other items
            for j in range(len(population)):
                # Check if our 'i' pint is dominated by out 'j' point
                if all(scores[j] <= scores[i]) and any(scores[j] < scores[i]):
                    # j dominates i. Label 'i' point as not on Pareto front
                    pareto_front[i] = 0
                    # Stop further comparisons with 'i' (no more comparisons needed)
                    break
        return population[pareto_front]

    def cal_crowding(self, scores):
        population_size = len(scores[:, 0])
        num_scores = len(scores[0, :])

        # create crowding matrix of population (row) and score (column)
        crowding_matrix = np.zeros((population_size, num_scores))

        # normalise scores (ptp is max-min)
        normed_scores = (scores - scores.min(0)) / scores.ptp(0)

        # calculate crowding distance for each score in turn
        for col in range(num_scores):
            crowding = np.zeros(population_size)

            # end points have maximum crowding
            crowding[0] = 1
            crowding[population_size - 1] = 1

            # Sort each score (to calculate crowding between adjacent scores)
            sorted_scores = np.sort(normed_scores[:, col])

            sorted_scores_index = np.argsort(normed_scores[:, col])

            # Calculate crowding distance for each individual
            crowding[1 : population_size - 1] = (
                sorted_scores[2:population_size]
                - sorted_scores[0 : population_size - 2]
            )

            # resort to orginal order (two steps)
            re_sort_order = np.argsort(sorted_scores_index)
            sorted_crowding = crowding[re_sort_order]

            # Record crowding distances
            crowding_matrix[:, col] = sorted_crowding

        # Sum crowding distances of each score
        crowding_distances = np.sum(crowding_matrix, axis=1)

        return crowding_distances

    def _append_pareto(self, pool, scores, more_number):
        second_pareto = self.identify_pareto(scores, pool)
        if len(second_pareto) >= more_number:
            crowding_distances = self.cal_crowding(scores)
            idx = [i for i in np.argsort(crowding_distances)[-more_number:]]
            append_pareto = np.array([pool[i] for i in idx])
        else:
            append_pareto = second_pareto
        return append_pareto

    def get_pareto_parents(self, pareto):
        pareto_pool = []
        for chromosome in self.population.tolist():
            if chromosome not in pareto.tolist():
                pareto_pool.append(chromosome)
        pareto_pool = np.array(pareto_pool)
        scores = self.fitness(pareto_pool)
        more_number = self.min_parents - len(pareto)
        if more_number != 0:
            append_pareto = self._append_pareto(pareto_pool, scores, more_number)
            if append_pareto is not None:
                pareto_parents = np.vstack((pareto, append_pareto))
            else:
                print("***Error: Pareto to append is none.")
        return pareto_parents

    def initial_population(self):
        population_size = (self.num_population, len(self.crane_location))
        print("Population size: ", population_size)
        initial_population = []
        while len(initial_population) < self.num_population:
            while 1:
                random_layout = self.random_layout()
                self.set_layout(random_layout)
                chromosome = self.get_chromosome()
                if self.check_layout() is True:
                    cost, _ = self.cal_function()
                    if math.isnan(cost) is False:
                        # print(
                        #     f"#{len(initial_population)+1} Crane layout: ", self.layout
                        # )
                        break
            initial_population.append(chromosome)
        self.initial_population = np.array(initial_population)
        self.set_population(self.initial_population)

    def evolve(self):
        print(f"GA starts evolving. {self.generation} Generations to go.")
        self.hist = []
        for _ in tqdm(range(self.generation)):
        # for i in range(self.generation):
            # print(f"Generation [{i+1}/{self.generation}]")
            pareto, pareto_fitness = self.evaluate()
            self.hist.append(pareto_fitness)
            parents = self.selection(pareto)
            children = []
            num_children = self.num_population - len(parents)
            while len(children) < num_children:
                children_crossover = self.crossover(parents, num_children)
                children_mutation = self.mutation(children_crossover)
                for child in children_mutation:
                    if len(children) >= num_children:
                        break
                    child_layout = self.chromosome_layout(child)
                    self.set_layout(child_layout)
                    if self.check_layout() is True:
                        cost, _ = self.cal_function()
                        if math.isnan(cost) is False:
                            children.append(child)
                    else:
                        continue
                    # newly added
                    # if len(children) >= num_children:
                    #     break
            children = np.array(children)

            self.population[0 : len(parents), :] = parents
            self.population[len(parents) :, :] = children
            self.set_population(self.population)

    def evaluate(self):
        fitness = self.fitness()
        pareto = self.identify_pareto(fitness)
        pareto_fitness = self.fitness(pool=pareto)
        return pareto, pareto_fitness

    def get_hist(self):
        hist_cost = []
        hist_time = []
        for best_outputs in self.hist:
            cost, time = best_outputs.mean(axis=0)
            hist_cost.append(cost)
            hist_time.append(time)
        hist_cost = np.array(hist_cost)
        hist_time = np.array(hist_time)
        return hist_cost, hist_time
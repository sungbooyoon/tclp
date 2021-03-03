import copy
import numpy as np
import json
import math
import random
from itertools import combinations
import pareto_ga_multicrane as ga

# get distance
def get_distance(location_1, location_2):
    distance_x = location_1[0] - location_2[0]
    distance_y = location_1[1] - location_2[1]
    distance = math.sqrt((distance_x ** 2 + distance_y ** 2))
    return distance


# get location data
def get_location():
    crane_location_list = []
    unit_location_list = []
    trailer_location_list = [[4, 6], [4, 10], [17, 1], [22, 4], [22, 9], [22, 14]]
    file_path = "data/simulation.json"
    with open(file_path, "r") as json_file:
        all_data = json.load(json_file)
        tc = all_data["tc_location_list"]
        unit = all_data["unit_location_list"]
        for i in tc:
            crane_location_list.append([i["x"], i["y"]])
        for i in unit:
            unit_location_list.append([i["x"], i["y"]])
    return crane_location_list, unit_location_list, trailer_location_list


crane_location_list, unit_location_list, trailer_location_list = get_location()


# get crane data
def get_cranedb(file_path):
    with open(file_path, "r") as json_file:
        all_data = json.load(json_file)
        crane_data = all_data["tc_list"]

        # Replace string values in dict to int or float
        for model in crane_data:
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
    return crane_data


file_path = "data/simulation.json"
crane_data = get_cranedb(file_path)


# get crane location and model
def get_crane(chromosome):
    crane_model = []
    crane_location_idx = np.where(chromosome != 0)[0]
    for i in crane_location_idx:
        crane_model_idx = chromosome[i] - 1
        crane_data[crane_model_idx]["x"] = crane_location_list[i][0]
        crane_data[crane_model_idx]["y"] = crane_location_list[i][1]
        crane_model.append(copy.copy(crane_data[crane_model_idx]))
    return np.array(crane_model)


def get_location_from_get_crane(chromosome):
    crane_location_idx = np.where(chromosome != 0)[0]
    location_list = []
    for i in crane_location_idx:
        location_list.append([crane_location_list[i][0], crane_location_list[i][1]])
    return location_list


# check distance
def check_distance(location_list, crane_model):
    for i in location_list:
        distance_list = []
        boolean_list = []
        for v in crane_model:
            distance_list.append(get_distance(i, [v["x"], v["y"]]))
        for idx in range(len(distance_list)):
            boolean_list.extend([distance_list[idx] <= crane_model[idx]["rmax"]])
        if any(boolean_list):
            continue
        else:
            return False
    return True


def check_crane_distance(crane_model):
    location_list = []
    for i in range(len(list(crane_model))):
        location_list.append([crane_model[i]["x"], crane_model[i]["y"]])
    combination_list = list(combinations(location_list, 2))
    for i in combination_list:
        distance = get_distance(i[0], i[1])
        if distance < 2:
            return False
    return True


# make random chromosome
def make_random_chromosome(crane_location_number, crane_number):
    chromosome = np.zeros(crane_location_number, int)
    crane_location_idx_list = random.sample(range(crane_location_number), crane_number)
    for x in range(len(crane_location_idx_list)):
        chromosome[crane_location_idx_list[x]] = (
            random.randrange(len(crane_data)) + 1
        )  # 실제 모델 넘버는 1부터 17!!!
    return chromosome


# check constraints
def check_chromosome(chromosome):
    crane_model = get_crane(chromosome)
    with open(file_path, "r") as json_file:
        all_data = json.load(json_file)
        crane_data = all_data["tc_list"]

        # Replace string values in dict to int or float
        for model in crane_data:
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
    building_height = 4.54
    # check distance
    if check_distance(unit_location_list, crane_model):
        pass
    else:
        return False
    # check height
    boolean_list = []
    for i in crane_model:
        if building_height <= i["height"]:
            continue
        else:
            boolean_list.append(1)
            break
    # check crane distan    ce
    if check_crane_distance(crane_model):
        pass
    else:
        return False

    try:
        cal_total_fee(chromosome)
        pass
    except:
        return False
    # check task allocation
    task_allocation_list = final_task_allocation(chromosome)
    for i in task_allocation_list:
        if i[1] != 0:
            continue
        else:
            boolean_list.append(1)
            break
    if len(boolean_list) > 0:
        return False
    else:
        return True


def task_allocation(chromosome):
    tasks = []
    task_allocation_list = []
    crane_model = get_crane(chromosome)
    for i in range(len(crane_model)):
        task_allocation_list.append([i, 0])
    unit_list = get_3d_unit_list()  # unit list is 3D here!!!!
    for i in unit_list:
        distance_list = []
        boolean_list = []
        for v in crane_model:
            distance_list.append(get_distance([i[0], i[1]], [v["x"], v["y"]]))
        for idx in range(len(distance_list)):
            boolean_list.extend([distance_list[idx] <= crane_model[idx]["rmax"]])
        if any(boolean_list):
            idx_list = [a for a, v in enumerate(boolean_list) if v is True]
            task_allocated_list = [task_allocation_list[a][1] for a in idx_list]
            task = idx_list[task_allocated_list.index(min(task_allocated_list))]
            tasks.append(task)
            task_allocation_list[task][1] += 1
            continue
    return tasks


def get_3d_unit_list():
    demand_height = [
        8.1,
        11.22,
        14.34,
        17.46,
        20.58,
        23.7,
        26.2,
        29.94,
        33.06,
        36.18,
        39.3,
    ]
    unit_location_3d_list = []
    for i in unit_location_list:
        for a in demand_height:
            unit_location_3d_list.append([i[0], i[1], a])
    return unit_location_3d_list


# calculate total time
def cal_total_fee(chromosome):
    supply_height = 0
    alpha = 0.25  # nadoushani
    beta = 1.0
    crane_model = get_crane(chromosome)
    fee_list = []
    unit_location_3d_list = get_3d_unit_list()
    tasks = task_allocation(chromosome)
    if len(tasks) != len(unit_location_3d_list):
        print("Error: tasks are not full")
    for i, v in enumerate(unit_location_3d_list):
        # tl = random.triangular(25, 20, 15) + random.triangular(20, 17, 15) + 7
        # tu = random.triangular(30, 20, 15) + random.triangular(30, 22, 15) + 7
        tl = 7
        tu = 7
        tv = abs(v[2] - supply_height) / crane_model[tasks[i]]["vv"]
        pd = get_distance(
            [v[0], v[1]], [crane_model[tasks[i]]["x"], crane_model[tasks[i]]["y"]]
        )
        ps = 3  # subject to change!!!!!!!
        lcr = random.uniform(
            abs(pd - ps) + 0.01, pd + ps - 0.01
        )  # subject to change!!!!!!!
        tr = abs(pd - ps) / crane_model[tasks[i]]["vr"]
        x = (-(lcr ** 2) + pd ** 2 + ps ** 2) / (2 * pd * ps)
        tw = 1 / ((crane_model[tasks[i]]["vw"]) * 10) * math.acos(x)
        th = max([tr, tw]) + alpha * min([tr, tw])
        t = (
            max([th, tv]) + beta * min([th, tv]) + tl + tu
        ) * 78 / 52 + 61.7  # 유닛 실제 개수/코딩 개수
        fee = t * crane_model[tasks[i]]["rentalcost"] * 1000 / 8 / 60
        fee_list.append(fee)
    fixed_cost = []
    for i in range(len(crane_model)):
        fixed_cost.append(crane_model[i]["fixedcost"])
    total_fee = sum(fee_list) + sum(fixed_cost) * 1000
    return total_fee * 10


# get collision factor
def get_collision_factor(crane_model):
    location_list = []
    r_max_list = []
    height_list = []
    shared_space = 0
    for i in range(len(list(crane_model))):
        location_list.append([crane_model[i]["x"], crane_model[i]["y"]])
        r_max_list.append(crane_model[i]["rmax"])
        height_list.append(crane_model[i]["height"])
    combination_list = list(combinations(location_list, 2))
    for i in combination_list:
        distance = get_distance(i[0], i[1])
        r1 = r_max_list[location_list.index(i[0])]
        r2 = r_max_list[location_list.index(i[1])]
        shared_space += cal_shared_space(distance, r1, r2)
    return shared_space


def cal_shared_space(distance, r1, r2):
    if distance < r1 + r2:
        a = r1 ** 2
        b = r2 ** 2
        x = (a - b + distance ** 2) / (2 * distance)
        z = x ** 2
        y = math.sqrt(abs(a - z))
        if distance <= abs(r1 - r2):
            return math.pi * min(a, b)
        return (
            10 * a * math.asin(y / r1)
            + b * math.asin(y / r2)
            - y * (x + math.sqrt(abs(z + b - a)))
        )
    else:
        return 0


def identify_pareto(scores, population):
    population_size = scores.shape[0]
    # Create a starting list of items on the Pareto front
    # All items start off as being labelled as on the Parteo front
    pareto_front = np.ones(population_size, dtype=bool)
    # Loop through each item. This will then be compared with all other items
    for i in range(population_size):
        # Loop through all other items
        for j in range(population_size):
            # Check if our 'i' pint is dominated by out 'j' point
            if all(scores[j] <= scores[i]) and any(scores[j] < scores[i]):
                # j dominates i. Label 'i' point as not on Pareto front
                pareto_front[i] = 0
                # Stop further comparisons with 'i' (no more comparisons needed)
                break
    # Return ids of scenarios on pareto front

    return population[pareto_front]


def calculate_crowding(scores):
    population_size = len(scores[:, 0])
    number_of_scores = len(scores[0, :])

    # create crowding matrix of population (row) and score (column)
    crowding_matrix = np.zeros((population_size, number_of_scores))

    # normalise scores (ptp is max-min)
    normed_scores = (scores - scores.min(0)) / scores.ptp(0)

    # calculate crowding distance for each score in turn
    for col in range(number_of_scores):
        crowding = np.zeros(population_size)

        # end points have maximum crowding
        crowding[0] = 1
        crowding[population_size - 1] = 1

        # Sort each score (to calculate crowding between adjacent scores)
        sorted_scores = np.sort(normed_scores[:, col])

        sorted_scores_index = np.argsort(normed_scores[:, col])

        # Calculate crowding distance for each individual
        crowding[1 : population_size - 1] = (
            sorted_scores[2:population_size] - sorted_scores[0 : population_size - 2]
        )

        # resort to orginal order (two steps)
        re_sort_order = np.argsort(sorted_scores_index)
        sorted_crowding = crowding[re_sort_order]

        # Record crowding distances
        crowding_matrix[:, col] = sorted_crowding

    # Sum crowding distances of each score
    crowding_distances = np.sum(crowding_matrix, axis=1)

    return crowding_distances


def append_pareto(pop, scores, more_number):
    second_pareto = identify_pareto(scores, pop)
    if len(second_pareto.tolist()) >= more_number:
        append_list = []
        crowding_distances = calculate_crowding(scores)
        index_list = [i for i in np.argsort(crowding_distances)[-more_number:]]
        for idx in index_list:
            append_list.append(pop.tolist()[idx])
        return append_list
    else:
        append_list = second_pareto.tolist()
        return append_list


def make_pareto_parents(parents_list, pop, min_number_parents):
    pareto_pool = []
    for i in pop.tolist():
        if i not in parents_list:
            pareto_pool.append(i)
    pareto_pool = np.array(pareto_pool)
    scores = ga.cal_pop_fitness(pareto_pool)
    more_number = min_number_parents - len(parents_list)
    append_list = append_pareto(pareto_pool, scores, more_number)
    for a in append_list:
        parents_list.append(a)
    return parents_list


def final_task_allocation(chromosome):
    tasks = []
    task_allocation_list = []
    crane_model = get_crane(chromosome)
    for i in range(len(crane_model)):
        task_allocation_list.append([i, 0])
    unit_list = get_3d_unit_list()  # unit list is 3D here!!!!
    for i in unit_list:
        distance_list = []
        boolean_list = []
        for v in crane_model:
            distance_list.append(get_distance([i[0], i[1]], [v["x"], v["y"]]))
        for idx in range(len(distance_list)):
            boolean_list.extend([distance_list[idx] <= crane_model[idx]["rmax"]])
        if any(boolean_list):
            idx_list = [a for a, v in enumerate(boolean_list) if v is True]
            task_allocated_list = [task_allocation_list[a][1] for a in idx_list]
            task = idx_list[task_allocated_list.index(min(task_allocated_list))]
            tasks.append(task)
            task_allocation_list[task][1] += 1
            continue
    return task_allocation_list


def get_total_operation_time(chromosome):
    supply_height = 0
    alpha = 0.25  # nadoushani
    beta = 1.0
    crane_model = get_crane(chromosome)
    time_list = []
    for i in range(len(crane_model)):
        time_list.append([i, 0])
    unit_location_3d_list = get_3d_unit_list()
    tasks = task_allocation(chromosome)
    if len(tasks) != len(unit_location_3d_list):
        print("Error: tasks are not full")
    for i, v in enumerate(unit_location_3d_list):
        # tl = random.triangular(25, 20, 15) + random.triangular(20, 17, 15) + 7
        # tu = random.triangular(30, 20, 15) + random.triangular(30, 22, 15) + 7
        tl = 7
        tu = 7
        tv = abs(v[2] - supply_height) / crane_model[tasks[i]]["vv"]
        pd = get_distance(
            [v[0], v[1]], [crane_model[tasks[i]]["x"], crane_model[tasks[i]]["y"]]
        )
        ps = 3  # subject to change!!!!!!!
        lcr = random.uniform(
            abs(pd - ps) + 0.01, pd + ps - 0.01
        )  # subject to change!!!!!!!
        tr = abs(pd - ps) / crane_model[tasks[i]]["vr"]
        x = (-(lcr ** 2) + pd ** 2 + ps ** 2) / (2 * pd * ps)
        tw = 1 / ((crane_model[tasks[i]]["vw"]) * 10) * math.acos(x)
        th = max([tr, tw]) + alpha * min([tr, tw])
        t = (
            (max([th, tv]) + beta * min([th, tv]) + tl + tu) * 78 / 52 + 61.7
        ) / 60  # 유닛 실제 개수/코딩 개수
        time_list[tasks[i]][1] += t
    return time_list


def get_max_operation_time(time_list):
    time_list_list = []
    for i in time_list:
        time_list_list.append(i[1])
    return max(time_list_list)
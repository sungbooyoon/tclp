import numpy as np
import random
from itertools import combinations
import openpyxl
import math
from pprint import pprint
import matplotlib.pyplot as plt
import pickle
import time
import os
from tqdm import tqdm

from modularclass import DataLoader, Crane
from modularclass import GeneticAlgorithms as GA

# Define file paths
simulation_file_path = "data/simulation.json"

# Data loader
data = DataLoader(simulation_file_path)
crane_location = data.get_crane_location()
building_info = data.get_building_info()
unit_location = data.get_unit_location()
crane_model = data.get_crane_model()
parameters = data.get_parameters()

computing_times = np.array([])
for _ in tqdm(range(20)):
    start = time.time()
    crane = Crane(crane_location, unit_location, crane_model, building_info, parameters)
    ga = GA(crane_location, unit_location, crane_model, building_info, parameters)

    ga.initial_population()
    ga.evolve()

    end = time.time()
    computing_times = np.load("result/no_mask/times.npy")
    computing_times = np.append(computing_times, end - start)
    np.save("result/no_mask/times.npy", computing_times)

# Define file paths
simulation_file_path = "data/simulation_mask1.json"

# Data loader
data = DataLoader(simulation_file_path)
crane_location = data.get_crane_location()
building_info = data.get_building_info()
unit_location = data.get_unit_location()
crane_model = data.get_crane_model()
parameters = data.get_parameters()

computing_times_mask = np.array([])
for _ in tqdm(range(20)):
    start = time.time()
    crane = Crane(crane_location, unit_location, crane_model, building_info, parameters)
    ga = GA(crane_location, unit_location, crane_model, building_info, parameters)

    ga.initial_population()
    ga.evolve()

    end = time.time()
    computing_times_mask = np.load("result/mask1/times.npy")
    computing_times_mask = np.append(computing_times_mask, end - start)
    np.save("result/mask1/times.npy", computing_times_mask)


# pareto, pareto_fitness = ga.evaluate()

# with open("result/no_mask/hist.pickle", "wb") as f:
#     pickle.dump(ga.hist, f)
# np.save("result/no_mask/pareto", pareto)
# np.save("result/no_mask/pareto_fitness", pareto_fitness)
# np.save("result/no_mask/population", ga.population)

# hist_cost, hist_time = ga.get_hist()

# plt.subplot(1, 2, 1), plt.plot(hist_cost)
# plt.subplot(1, 2, 2), plt.plot(hist_time)
# plt.show()

# print("Time: ", end - start)
# with open("result/no_mask/time.pickle", "wb") as f:
#     pickle.dump(end - start, f)

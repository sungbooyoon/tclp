import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from modularclass import DataLoader, Crane
from modularclass import GeneticAlgorithms as GA


# Define file paths
simulation_file_path = "data/simulation_mask1.json"

# Data loader
data = DataLoader(simulation_file_path)
crane_location = data.get_crane_location()
building_info = data.get_building_info()
unit_location = data.get_unit_location()
crane_model = data.get_crane_model()
parameters = data.get_parameters()

crane = Crane(crane_location, unit_location, crane_model, building_info, parameters)
ga = GA(crane_location, unit_location, crane_model, building_info, parameters)


with open("result/no_mask/hist.pickle", "rb") as f:
    hist = pickle.load(f)
with open("result/mask1/time_mask1.pickle", "rb") as f:
    time = pickle.load(f)
pareto = np.load("result/mask1/pareto_mask1.npy")
pareto_fitness = np.load("result/mask1/pareto_fitness_mask1.npy")
population = np.load("result/mask1/population_mask1.npy")

for i in pareto:
    layout = crane.chromosome_layout(i)
    crane.set_layout(layout)
#     print(crane.get_locations())
#     print(crane.layout)

# print(pareto_fitness)
# print(time)

times_no_mask = np.load("result/no_mask/times.npy")
times_mask1 = np.load("result/mask1/times.npy")
# print(times_no_mask.mean())
# print(times_mask1.mean())
# plt.plot(times_no_mask, ".", color="black", label="No mask")
# plt.plot(times_mask1, ".", color="cornflowerblue", label="Mask")
# plt.axhline(times_no_mask.mean(), linestyle="dashed", color="black", linewidth=0.5)
# plt.axhline(times_mask1.mean(), linestyle="dashed", color="cornflowerblue", linewidth=0.5)
# plt.xlabel('Iteration')
# plt.ylabel('Computing Time')
# plt.legend()
# plt.show()
"""
min_cost = []
min_duration = []
avg_cost = []
avg_duration = []
for i in hist:
    i = pd.DataFrame(i)
    i = i.dropna(axis=0, how="any", thresh=None)
    i = np.array(i)
    min_cost.append(np.min(i, axis=0)[0])
    min_duration.append(np.min(i, axis=0)[1])
    avg_cost.append(np.average(i, axis=0)[0])
    avg_duration.append(np.average(i, axis=0)[1])

min_cost = np.array(min_cost)
min_duration = np.array(min_duration)
avg_cost = np.array(avg_cost)
avg_duration = np.array(avg_duration)

gen = np.arange(1, 201)
plt.subplot(121)
plt.title('Cost')
plt.xlabel('Generation')
plt.ylabel('Fitness value')
plt.plot(gen, min_cost)

plt.subplot(122)
plt.title('Duration')
plt.xlabel('Generation')
plt.ylabel('Fitness value')
plt.plot(gen, min_duration)
plt.show()
"""

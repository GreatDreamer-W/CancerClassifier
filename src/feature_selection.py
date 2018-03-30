from random import sample

import numpy as np
from deap import creator, base, tools, algorithms
from deap.tools import selTournament

from src.dbi import dbi


def evaluate(individual, cluster1, cluster2):
    index = dbi(cluster1[:, individual], cluster2[:, individual])
    return index,


def get_clusters(data, label):
    size = label.size
    cluster_label = np.unique(label)
    cluster1_index = []
    cluster2_index = []
    for i in range(size):
        if label[i] == cluster_label[0]:
            cluster1_index.append(i)
        elif label[i] == cluster_label[1]:
            cluster2_index.append(i)
    cluster1 = data[cluster1_index]
    cluster2 = data[cluster2_index]
    return cluster1, cluster2


def genetic_algorithm(individual_size, n_gen, train_data, train_label):
    cluster1, cluster2 = get_clusters(train_data, train_label)
    individual_range = train_data.shape[1]

    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("indices", sample, range(individual_range), individual_size)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual, )
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutUniformInt, low=0, up=individual_range-1, indpb=0.1)
    toolbox.register("select", selTournament, tournsize=3)
    toolbox.register("evaluate", evaluate, cluster1=cluster1, cluster2=cluster2)

    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("min_index", np.argmin)
    stats.register("max", np.max)
    stats.register("max_index", np.argmax)

    pop = toolbox.population(n=50)
    final_population, log_book = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.1, ngen=n_gen, stats=stats,
                                                     verbose=True)
    return final_population, log_book

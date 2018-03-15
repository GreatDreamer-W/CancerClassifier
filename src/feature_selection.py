from random import sample

import numpy as np
from deap import creator, base, tools, algorithms
from sklearn.svm import SVC

from src.classifier import svm
from src.dbi import dbi


def evaluate(individual, cluster1, cluster2):
    print("------------------我是一条分割线-------------------")
    print(individual.shape)
    return dbi(cluster1[:, individual], cluster2[:, individual])


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


def genetic_algorithm(individual_size, train_data, train_label):
    cluster1, cluster2 = get_clusters(train_data, train_label)
    individual_range = train_data.shape[1]

    creator.create("FitnessMin", base.Fitness, weights=(0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("indices", sample, range(individual_range), individual_size)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual, )
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutUniformInt, 0, individual_range, 0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", evaluate, cluster1, cluster2)

    ind1 = toolbox.individual()
    ind2 = toolbox.individual()
    child1, child2 = [toolbox.clone(ind) for ind in (ind1, ind2)]
    tools.cxTwoPoint(child1, child2)
    print(ind1)
    print(ind2)
    print(child1)
    print(child2)


    selected = tools.selBest([child1, child2, ind1, ind2], 2)
    print(child1 in selected)
    print(child2 in selected)


    # pop = toolbox.population(n=5)
    # algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.1, ngen=5)


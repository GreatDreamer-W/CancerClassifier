from random import sample

from deap import creator, base, tools


def genetic_algorithm(individual_range, individual_size):
    creator.create("FitnessMin", base.Fitness, weights=(0.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("indices", sample, range(individual_range), individual_size)
    toolbox.register("individual", tools.initIterate(), creator.Individual, toolbox.indices)
    toolbox.register("population", tools.initRepeat(), list, toolbox.indivisual)
    toolbox.register("mate", tools.cxTwoPoint())
    toolbox.register("mutate", tools.mutGaussian(), )



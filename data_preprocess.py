import random
from functools import partial

import pandas
from deap import creator, base, tools
from sklearn.preprocessing import MinMaxScaler


def data_preprocessing():
    train_data = pandas.read_csv("C:/Users/Yirui/OneDrive/SRTP/train.csv")
    scaler = MinMaxScaler(feature_range=(0, 1000))
    scaler.fit(train_data)
    print(scaler.transform(train_data))


data_preprocessing()
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("FitnessMulti", base.Fitness, weights=(-1.0, 1.0))
gen_idx = partial(random.sample, range(10), 10)
print(gen_idx)

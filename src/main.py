import numpy

from src.classifier import svm, knn
from src.data_preprocess import data_preprocess
from src.feature_selection import get_clusters

train_data, train_label, test_data, test_label = data_preprocess()
data = numpy.vstack((train_data, test_data))
target = numpy.hstack((train_label, test_label))
cluster1, cluster2 = get_clusters(data, target)

# dbi_array = []
# for i in range(cluster1.shape[1]):
#     dbi_array.append(dbi(cluster1[:, i], cluster2[:, i]))
#
# sorted_dbi = sorted(dbi_array)
# minmax = sorted_dbi[50]

index_array = [757,  759,  803,  1143, 1629, 1684, 1744, 1778, 1828, 1833,
               1881, 1908, 2019, 2120, 2127, 2287, 2334, 2353, 2401, 2440,
               2496, 2641, 3251, 4081, 4106, 4166, 4195, 4210, 4228, 4290,
               4327, 4365, 4372, 4376, 4679, 4846, 4893, 4972, 5170, 5500,
               5687, 5771, 6040, 6224, 6280, 6375, 6377, 6622, 6853, 7117]

# for i in range(cluster1.shape[1]):
#     if dbi_array[i] < minmax:
#         index_array.append(i)

print(data.shape, target.shape)
svm(data[:, index_array], target)

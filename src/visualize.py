import matplotlib.pyplot as plt
import numpy

from src.data_preprocess import data_preprocess

index_array = [757,  759,  803,  1143, 1629, 1684, 1744, 1778, 1828, 1833,
               1881, 1908, 2019, 2120, 2127, 2287, 2334, 2353, 2401, 2440,
               2496, 2641, 3251, 4081, 4106, 4166, 4195, 4210, 4228, 4290,
               4327, 4365, 4372, 4376, 4679, 4846, 4893, 4972, 5170, 5500,
               5687, 5771, 6040, 6224, 6280, 6375, 6377, 6622, 6853, 7117]

train_data, train_label, test_data, test_label = data_preprocess()
data = numpy.vstack((train_data, test_data))
target = numpy.hstack((train_label, test_label))

data = data[:, index_array]

# 冒泡排序
for i in range(50):
    for j in range(50-i-1):
        if numpy.mean(data[47:, j]) < numpy.mean(data[47:, j+1]):
            data[:, [j, j+1]] = data[:, [j+1, j]]

# 交换部分行
data[:, [22, 23]] = data[:, [23, 22]]
data[:, [23, 28]] = data[:, [28, 23]]

data = numpy.vstack([data[:20, :], data[47:61, :], data[34:47, :],
                     data[20:34, :], data[61:, :]]).transpose()
target = numpy.hstack([target[:20], target[47:61], target[34:47],
                       target[20:34], target[61:]])

plt.figure(figsize=(12, 8))
result = plt.imshow(data, plt.get_cmap('cool'))
plt.colorbar(result)
plt.show()

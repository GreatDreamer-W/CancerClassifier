# CancerClassifier
The project focuses on cancer classification based on DNA microarray datasets.
使用普林斯顿大学的62个样本的结肠癌数据与麻省理工学院72个样本的白血病数据。
在对数据进行预处理后，使用启发式搜索算法提取特征基因，采用聚类中DB指数作为遗传算法的评价函数，从数千基因中选择出50个特征基因，作为进一步分类的基础。
使用支持向量机作为分类器，由于样本数量较小，采用交叉验证发评价分类效果，
经过特征选择后，两个数据集均可达到较高的正确率。

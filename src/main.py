from src.classifier import svm, knn
from src.data_preprocess import data_preprocess
from src.feature_selection import get_clusters, genetic_algorithm, evaluate

train_data, train_label, test_data, test_label = data_preprocess()
genetic_algorithm(10, train_data, train_label)


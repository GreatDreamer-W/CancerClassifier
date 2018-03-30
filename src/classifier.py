import numpy
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


# def svm(train_data, train_label, test_data, test_label):
#     svc = SVC(kernel="poly")
#     svc.fit(train_data[:], train_label)
#     return svc.score(test_data.reshape(1, -1), test_label.reshape(1, -1))


def svm(data, target):
    svc = SVC(kernel="sigmoid")
    scores = cross_val_score(svc, data, target, cv=72)
    print(scores)
    print(numpy.mean(scores))


def knn(train_data, train_label, test_data, test_label, attributes):
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(train_data[:, attributes], train_label)
    print(knn.score(train_data[:, attributes], train_label))
    print(knn.score(test_data[:, attributes], test_label))

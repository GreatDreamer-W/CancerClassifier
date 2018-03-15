from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


def svm(train_data, train_label, test_data, test_label):
    svc = SVC()
    svc.fit(train_data, train_label)
    distance = svc.decision_function(train_data)
    print(svc.score(train_data, train_label))
    sum = 0
    count = 0
    for i in distance:
        if i > 0:
            count += 1
            sum += i
    print(sum / count)
    sum = 0
    count = 0
    for i in distance:
        if i < 0:
            count += 1
            sum += i
    print(sum / count)
    print(distance)
    # print(svc.get_params())
    # return test_result


def knn(train_data, train_label, test_data):
    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(train_data, train_label)
    test_result = neigh.predict(test_data)
    return test_result

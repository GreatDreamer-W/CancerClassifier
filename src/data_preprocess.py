from numpy import loadtxt, savetxt
from sklearn.preprocessing import scale


def data_preprocess():
    train_data = loadtxt("data/colon/train.csv", delimiter=",", encoding="utf-8-sig")
    train_label = loadtxt("data/colon/train_label.csv", delimiter=",", dtype="str", encoding="utf-8-sig")
    test_data = loadtxt("data/colon/test.csv", delimiter=",", encoding="utf-8-sig")
    test_label = loadtxt("data/colon/test_label.csv", delimiter=",", dtype="str", encoding="utf-8-sig")
    train_data = scale(train_data)
    test_data = scale(test_data)

    return train_data, train_label, test_data, test_label

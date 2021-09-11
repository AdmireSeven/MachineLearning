from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, LeaveOneOut
import matplotlib.pyplot as plt

iris = datasets.load_iris()

X = iris.data
y = iris.target
loo = LeaveOneOut()
# knn = KNeighborsClassifier(15)   # 默认k=5
# correct = 0

# for train, test in loo.split(X):
#     # print("留一划分：%s %s" % (train.shape, test.shape))
#     # X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.4, random_state=0)
#     knn.fit(X[train], y[train])
#     y_sample = knn.predict(X[test])
#     if y_sample == y[test]:
#         correct += 1
#
# print('Test accuracy:', correct/len(X))
K = []
Accuracy = []
for k in range(1, 16):
    correct = 0
    knn = KNeighborsClassifier(k)
    for train, test in loo.split(X):
        #print("留一划分：%s %s" % (train.shape, test.shape))
        #sX_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.4, random_state=0)
        #train_test_split 用于任意划分训练集与测试集 test_size是测试集百分比
        knn.fit(X[train], y[train])
        y_sample = knn.predict(X[test])
        if y_sample == y[test]:
            correct += 1
    K.append(k)
    Accuracy.append(correct/len(X))
    plt.plot(K, Accuracy)
    plt.xlabel('Accuracy')
    plt.ylabel('K')
    print('K=',k,'时分类精度为:',correct/len(X))

plt.show()
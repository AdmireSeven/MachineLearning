import random
from KnnClassify import *
#from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_circles

fig = plt.figure(1)
x1, y1 = make_circles(n_samples=400, factor=0.5, noise=0.1)     # 生成二维数组
# n_samples：生成样本数，内外平分   noise：异常点的比例   factor：内外圆之间的比例因子 ∈(0,1)
# x1[:,0]表示取所有坐标的第一维数据   x1[0,:]表示第一个坐标的所有维数据


k = 15
clf = KNN(k=k)
clf.fit(x1, y1);
# # 进行预测
x2 = random.random()
y2 = random.random()
X_sample = np.array([[x2, y2]])
y_sample = clf.predict(X_sample)
neighbors = clf.kneighbors(X_sample, return_distance=False)

plt.subplot(121)
plt.title('data by make_circles()')
colors = []
for i in y1:
    if i == 0:
        colors.append('b')
    else:
        colors.append('c')
plt.scatter(x1[:,0],x1[:,1],marker = 'o',c=colors)
plt.scatter(x2, y2, marker='*',c='r')

plt.subplot(122)
plt.title('make_circles 2')
plt.scatter(x1[:, 0], x1[:, 1], marker='o', c=colors)
plt.scatter(x2, y2, marker='*', c='r')
for i in neighbors[0]:
    #plt.plot([x1[i][0], X_sample[0][0]], [x1[i][1], X_sample[0][1]], '-', linewidth=0.6, c='b')
    plt.scatter([x1[i][0], X_sample[0][0]], [x1[i][1], X_sample[0][1]], marker='o', c='r')


plt.show()
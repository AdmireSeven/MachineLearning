"""
@Admire

Function：Project3_1

Email:admireseven@163.com
"""
from sklearn.datasets import make_circles, make_moons , make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score,normalized_mutual_info_score,adjusted_rand_score
from scipy.optimize import linear_sum_assignment as linear_assignment
import numpy as np

def cluster_acc(y_true, y_pred):

    y_true = np.array(y_true).astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_assignment(w.max() - w)
    ind = np.array(ind)
    ind = np.transpose(ind)
    return sum([w[i, j] for i, j in  ind]) * 1.0 / y_pred.size

fig = plt.figure(figsize=(8,7))
X1,y1 = make_circles(n_samples = 400, factor=0.3, noise=0.1)
plt.subplot(321)
plt.title('original')
colors = []
for i in y1:
    if i == 0:
        colors.append('b')
    else:
        colors.append('c')
plt.scatter(X1[:,0],X1[:,1],marker = 'o',linewidths=0.1,c=colors)
# x1[:,0]表示取所有坐标的第一维数据   x1[0,:]表示第一个坐标的所有维数据

plt.subplot(322)
plt.title('K-means')
kms = KMeans(n_clusters = 2, max_iter = 400)
                # n_cluster聚类中心数 max_iter迭代次数
y1_sample = kms.fit_predict(X1,y1) #计算并预测样本类别
centroids = kms.cluster_centers_
colors = []
for i in y1_sample:
    if i == 0:
        colors.append('b')
    else:
        colors.append('c')
plt.scatter(X1[:,0],X1[:,1],marker = 'o',linewidths=0.1,c=colors)
# x1[:,0]表示取所有坐标的第一维数据   x1[0,:]表示第一个坐标的所有维数据
plt.scatter(centroids[:,0],centroids[:,1],marker = '*',linewidths=0.5,c='r')

X2,y2 = make_moons(n_samples = 400, noise=0.1)
plt.subplot(323)
colors = []
for i in y2:
    if i == 0:
        colors.append('b')
    else:
        colors.append('c')
plt.scatter(X2[:,0],X2[:,1],marker = 'o',linewidths=0.1,c=colors)
# x1[:,0]表示取所有坐标的第一维数据   x1[0,:]表示第一个坐标的所有维数据

plt.subplot(324)
kms = KMeans(n_clusters = 2, max_iter = 400)
                # n_cluster聚类中心数 max_iter迭代次数
y2_sample = kms.fit_predict(X2,y2) #计算并预测样本类别
centroids = kms.cluster_centers_
colors = []
for i in y2_sample:
    if i == 0:
        colors.append('b')
    else:
        colors.append('c')
plt.scatter(X2[:,0],X2[:,1],marker = 'o',linewidths=0.1,c=colors)
# x1[:,0]表示取所有坐标的第一维数据   x1[0,:]表示第一个坐标的所有维数据
plt.scatter(centroids[:,0],centroids[:,1],marker = '*',linewidths=0.5,c='r')

X3,y3 = make_blobs(n_samples = 1000, random_state=9)
plt.subplot(325)
colors = []
for i in y3:
    if i == 0:
        colors.append('b')
    else:
        colors.append('c')
plt.scatter(X3[:,0],X3[:,1],marker = 'o',linewidths=0.1,c=colors)
# x1[:,0]表示取所有坐标的第一维数据   x1[0,:]表示第一个坐标的所有维数据

plt.subplot(326)
kms = KMeans(n_clusters = 3, max_iter = 1000)
                # n_cluster聚类中心数 max_iter迭代次数
y3_sample = kms.fit_predict(X3,y3) #计算并预测样本类别
centroids = kms.cluster_centers_
colors = []
for i in y3_sample:
    if i == 0:
        colors.append('b')
    else:
        colors.append('c')
plt.scatter(X3[:,0],X3[:,1],marker = 'o',linewidths=0.1,c=colors)
# x1[:,0]表示取所有坐标的第一维数据   x1[0,:]表示第一个坐标的所有维数据
plt.scatter(centroids[:,0],centroids[:,1],marker = '*',linewidths=0.5,c='r')
plt.show()

ACC1 = cluster_acc(y1,y1_sample)
ACC2 = cluster_acc(y2,y2_sample)
ACC3 = cluster_acc(y3,y3_sample)

NMI1 = normalized_mutual_info_score(y1,y1_sample)
NMI2 = normalized_mutual_info_score(y2,y2_sample)
NMI3 = normalized_mutual_info_score(y3,y3_sample)

ARI1 = adjusted_rand_score(y1,y1_sample)
ARI2 = adjusted_rand_score(y2,y2_sample)
ARI3 = adjusted_rand_score(y3,y3_sample)

print('data by make_circles ACC = %.6f, NMI = %.6f, ARI = %.6f'%(ACC1,NMI1,ARI1))
print('data by make_moons ACC = %.6f, NMI = %.6f, ARI = %.6f'%(ACC2,NMI2,ARI2))
print('data by make_blobs ACC = %.1f, NMI = %.1f, ARI = %.1f'%(ACC3,NMI3,ARI3))

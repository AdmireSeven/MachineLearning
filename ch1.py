from random import random

from sklearn.datasets import make_circles
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure(1)
x1,y1= make_circles(n_samples=800, factor=0.5, noise=0.1)
plt.subplot(121)
plt.title('make_circles function example')
plt.scatter(x1[:, 0], x1[:, 1], marker='o', c=y1)

plt.subplot(122)
x2, y2 = make_moons(n_samples=800, noise=0.1)
plt.title('make_moons function example')
plt.scatter(x2[:, 0], x2[:, 1], marker='o', c=y2)
plt.show()
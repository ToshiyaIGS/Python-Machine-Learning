# coding: utf-8
import pandas as pd
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
#目的変数
y = df.iloc[:100, 4].values
import numpy as np
y = np.where(y=='Iris-setosa', -1, 1)
X = df.iloc[:100, [0, 2]].values
from Perceptron import Perceptron
ppn = Perceptron(eta=0.1, n_iter=10)
ppn.fit(X, y)
import matplotlib.pyplot as plt
plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
plt.xlabel('Epopchs')
plt.ylabel('Number of update')
plt.show()
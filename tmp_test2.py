# coding: utf-8
import pandas as pd
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
#目的変数
y = df.iloc[:100, 4].values
import numpy as np
y = np.where(y=='Iris-setosa', -1, 1)
X = df.iloc[:100, [0, 2]].values
X_std = np.copy(X)
X_std[:, 0] = (X[:,0] - X[:,0].mean()) / X[:, 0].std()
X_std[:, 1] = (X[:,1] - X[:,1].mean()) / X[:, 1].std()
from AdalineSGD import AdalineSGD
ada = AdalineSGD(n_iter=15, eta=0.01, random_state=1)
ada.fit(X_std, y)
from plot_decision_regions import *
plot_decision_regions(X_std, y, classifier=ada)
import matplotlib.pyplot as plt
plt.title('Adaline - Stohastice Gradient Descent')
plt.xlabel('sepal length [standardized]')
plt.ylabel('petal length [standardized]')
plt.legend(loc='upper left)
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()
plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Average Cost')
plt.show()
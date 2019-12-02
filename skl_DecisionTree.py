from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from plot_decision_regions2 import plot_decision_regions2
import matplotlib.pyplot as plt
import numpy as np

# データの読み込み
iris = datasets.load_iris()

# 説明変数・目的変数の特定
X = iris.data[:, [2, 3]]
y = iris.target

# 訓練データ・検証データの分割
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1, stratify=y
)

# 訓練データと特徴データを結合
X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))

# ジニ不純度を指標とする決定木のインスタンスを生成
tree = DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=1)

# 決定木のモデルをトレーニングデータに適合させる
tree.fit(X_train, y_train)
plot_decision_regions2(X_combined, y_combined, classifier=tree, test_idx=range(105, 150))

plt.xlabel('petal length (cm)')
plt.ylabel('petal width (cm)')
plt.tight_layout()
plt.show()

from pydotplus import graph_from_dot_data
from sklearn.tree import export_graphviz
dot_data = export_graphviz(tree, filled=True, rounded=True, class_names=['Setosa', 'Versicolor', 'Virginics'], feature_names=['petal length', 'petal width'], out_file=None)
graph = graph_from_dot_data(dot_data)
graph.write_png('tree.png')
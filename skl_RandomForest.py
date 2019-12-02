from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
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

# ジニ不純度を指標とするランダムフォレストのインスタンスを生成
forest = RandomForestClassifier(criterion='gini', n_estimators=25, random_state=1, n_jobs=2)

# トレーニングデータにランダムフォレストのモデルを適合させる
forest.fit(X_train, y_train)

# プロット
plot_decision_regions2(X_combined, y_combined, classifier=forest, test_idx=range(105, 150))
plt.xlabel('petal length [cm]')
plt.ylabel('petal width [cm]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()
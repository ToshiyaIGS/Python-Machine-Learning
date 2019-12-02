from sklearn import datasets
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from plot_decision_regions2 import plot_decision_regions2
import matplotlib.pyplot as plt
import numpy as np

#データの読み込み
iris = datasets.load_iris()

#説明変数・目的変数の特定
X = iris.data[:, [2, 3]]
y = iris.target

#訓練データ・検証データの分割
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1, stratify=y
)

#データの標準化
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

#訓練データと特徴データを結合
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

#SVMモデルの構築・学習
svm = SVC(kernel='linear', C=1.0, random_state=1)
svm.fit(X_train_std, y_train)

#決定領域をプロット
plot_decision_regions2(X_combined_std, y_combined, classifier=svm, test_idx=range(105, 150))

#軸のラベル・凡例設定、グラフを表示
plt.xlabel('petal length (standardized)')
plt.ylabel('petal width (standardized)')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

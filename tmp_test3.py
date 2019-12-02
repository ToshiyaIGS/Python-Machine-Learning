# coding: utf-8
from sklearn import datasets
import numpy as np
# Irisデータセットをロード
iris = datasets.load_iris()
# 3, 4行目の特徴量を抽出
X = iris.data[:, 2:4]
X.shape
# クラスラベルを取得
y = iris.target
type(iris)
# 一意なクラスラベルを出力
print('Class labels:', np.unique(y))
from sklearn.model_selection import train_test_split
# トレーニングデータとテストデータに分割
# 全体の30%をテストデータにする
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1, stratify=y
)
print('Label counts in y', np.bincount(y))
print('Label counts in y_train', np.bincount(y_train))
print('Label counts in y_test', np.bincount(y_test))

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
# トレーニングデータの平均と標準偏差を計算
sc.fit(X_train)
# 平均と標準偏差を用いて標準化
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

from sklearn.linear_model import Perceptron
# エポック数40、学習率0.1でパーセプトロンのインスタンスを生成
ppn = Perceptron(n_iter=40, eta0=0.1, random_state=1)
# トレーニングデータをモデルに適合させる
ppn.fit(X_train_std, y_train)

# テストデータで予測を実施
y_pred = ppn.predict(X_test_std)
# ご分類のサンプルの個数を表示
print('Misclassified samples: %d' % (y_test != y_pred).sum())

from sklearn.metrics import accuracy_score
# 分類の正解率を表示
print('Accuracy score: %.2f' % accuracy_score(y_test, y_pred))

# トレーニングデータとテストデータの特徴量を行方向に結合
X_combined_std = np.vstack((X_train_std, X_test_std))
# トレーニングデータとテストデータのクラスラベルを結合
y_combined = np.hstack((y_train, y_test))
# 決定境界のプロット
from plot_decision_regions2 import plot_decision_regions2
plot_decision_regions2(X=X_combined_std, y=y_combined, classifier=ppn,
                    test_idx=range(105, 150))
# 軸のラベル設定
import matplotlib.pyplot as plt
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
# 凡例の設定（左上に設定）
plt.legend(loc='upper left')
# グラフを表示
plt.tight_layout()
plt.show()
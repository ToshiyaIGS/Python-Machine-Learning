from sklearn import datasets
from sklearn.linear_model import LogisticRegression
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

#ロジスティック回帰モデルの作成・学習
lr = LogisticRegression(C=100, random_state=1)
lr.fit(X_train_std, y_train)


#訓練データと特徴データを結合
# print(X_train_std.shape)
# print(X_test_std.shape)
# print(y_train.shape)
# print(y_test.shape)

X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

# print(X_combined_std.shape)
# print(y_combined.shape)

#決定領域をプロット
plot_decision_regions2(X_combined_std, y_combined, classifier=lr, test_idx=range(105, 150))

#軸のラベル・凡例設定、グラフを表示
plt.xlabel('petal length (standardized)')
plt.ylabel('petal width (standardized)')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

#最初の3つのサンプルの各ラベルへの所属確率を表示
print(lr.predict_proba(X_test_std[:3, :]))

#最初の3つのサンプルの各ラベル予測値を表示
print(lr.predict_proba(X_test_std[:3, :]).argmax(axis=1))

"""以下で正則化の強さを可視化する"""
#空のリストを作成（重み係数、逆正則化パラメータ）
weights, params = [], []

#10個の逆正則化パラメータに対応するロジスティック回帰モデルをそれぞれ処理
for c in np.arange(-5, 5):
    lr = LogisticRegression(C=10. ** c, random_state=1)
    lr.fit(X_train_std, y_train)
    #重み係数を格納
    weights.append(lr.coef_[1])
    #逆正則化パラメータを格納
    params.append(10. ** c)

#重み係数をNumpy配列に変換
weights = np.array(weights)

#横軸に逆正則化パラメータ、縦軸に重み係数を格納
plt.plot(params, weights[:, 0], label='petal length')
plt.plot(params, weights[:, 1], linestyle='--', label='petal width')
plt.xlabel('C')
plt.ylabel('weight coefficient')
plt.legend(loc='upper left')

plt.xscale('log')
plt.show()
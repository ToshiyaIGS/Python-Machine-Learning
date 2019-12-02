import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from plot_decision_regions2 import plot_decision_regions2

# 乱数シードを指定
np.random.seed(1)

# 標準正規分布に従う乱数で200行2列の行列を生成
X_xor = np.random.randn(200, 2)

# 2つの引数に対して排他的論理和を実行
y_xor = np.logical_xor(X_xor[:, 0] > 0, X_xor[:, 1] > 0)

# 排他的論理和の値が真の場合は1、魏の場合は-1を割り当てる
y_xor = np.where(y_xor, 1, -1)

# ラベル1を青のxでプロット
plt.scatter(X_xor[y_xor==1, 0], X_xor[y_xor==1, 1], c='b', marker='x', label=1)

# ラベル-1を赤の四角でプロット
plt.scatter(X_xor[y_xor==-1, 0], X_xor[y_xor==-1, 1], c='r', marker='s', label=-1)

# 軸の範囲を指定
plt.xlim([-3, 3])
plt.ylim([-3, 3])
plt.legend(loc='best')
plt.tight_layout()
plt.show()

# RBFカーネルによるSVMのインスタンスを生成
svm = SVC(kernel='rbf', random_state=1, gamma=0.10, C=10.0)
svm.fit(X_xor, y_xor)
plot_decision_regions2(X_xor, y_xor, classifier=svm)
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()
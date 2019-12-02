import numpy as np


class LogistgicRegressionGD(object):
    """　勾配降下法に基づくロジスティック回帰分類器

    パラメータ
    ------------
    eta : float
        学習率 (0.0 より大きく1.0 以下の値）
    n_iter : int
        トレーニングデータのトレーニング回数
    random_state : int
        重みを初期化するための乱数シード

    属性
    ------------
    w_ : 1次元配列
        適合後の重み
    cost_ : リスト
        各エポックでの誤差平方和のコスト関数

    """

    def __init__(self, eta=0.05, n_iter=100, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """トレーニングデータに適合させる

        パラメータ
        ------------
        X : { 配列のようなデータ構造 }, shape = {n_samples, n_features}
            トレーニングデータ
            n_samplesはサンプルの個数、n_featuresは特徴量の個数
        y : 配列のようなデータ構造, shape = {n_samples}
            目的変数

        戻り値
        ------------
        self : object

        """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.cost_ = []

        for _ in range(self.n_iter):  # トレーニング回数分トレーニングデータを反復
            net_input = self.net_input(X)
            output = self.activation(net_input)
            # 誤差 yi - Φ(z(i))の計算
            errors = (y - output)
            # 重み w1,w2,...,wmの更新 Δwi = ηΣ(y(i)-Φ(z(i)))xj(i)
            self.w_[1:] += self.eta * X.T.dot(errors)
            # 重み w0の更新　Δw0 = ηΣ(y(i)-Φ(z(i)))
            self.w_[0] += self.eta * errors.sum()
            # コスト関数の計算 J(w) = - Σ(y(i) * log(Φ(z(i)) + (1 - y(i)) * log (1 - φ(z(i))))
            cost = - y.dot(np.log(output)) - ((1 - y).dot(np.log(1- output)))
            # コストの格納
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        """総入力を計算"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        """線形活性化関数の出力を計算"""
        return 1. / (1. + np.exp(-np.clip(X, -250, 250)))

    def predict(self, X):
        """1ステップ後のクラスラベルを渡す"""
        return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)

import numpy as np
from scipy.sparse.linalg import svds

class PCA:
    def __init__(self, n_components, tol=0.0, random_seed=0):
        self.n_components = n_components #n_components: 次元圧縮後の次元数
        self.tol = tol #tol: トレランス、どれぐらいの計算誤差を許容するか
        self.random_state_ = np.random.RandomState(random_seed)

    def fit(self, X):
        v0 = self.random_state_.randn(min(X.shape)) #v0: SVDの計算に与える初期値
        xbar = X.mean(axis=0) #Xを縦方向に平均をとりxbarに格納
        Y = X - xbar #Xの各行から平均値xbarを引いたものをYに格納
        S = np.dot(Y.T, Y) #共分散行列を求めてSに格納
        U, Sigma, VT = svds(S, k=self.n_components, tol=self.tol, v0=v0)
        self.VT_ = VT[::-1, :]

    def transform(self, X):
        return self.VT_.dot(X.T).T

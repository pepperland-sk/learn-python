import numpy as np
from scipy import linalg

class LinearRegression:
    def __init__(self):
        self.w_ = None #w_格納される計算結果

    def fit(self, X, t):
        Xtil = np.c_[np.ones(X.shape[0]), X] #np.r_(): 行列を行方向に連結　x~を作成
        A = np.dot(Xtil.T, Xtil) #np.dot(): 行列の積を計算
        b = np.dot(Xtil.T, t)
        self.w_ = linalg.solve(A, b)

    def predict(self, X):
        if X.ndim == 1:
            X = X.reshape(1, -1)
        Xtil = np.c_[np.ones(X.shape[0]), X]
        return np.dot(Xtil, self.w_)

import numpy as np

def soft_thresholding(x, y):
    return np.sign(x) * max(abs(x) - y, 0)

class Lasso:
    def __init__(self, lambda_, tol=0.0001, max_iter=1000): #tol: 収束判定のためのトレランス（許容度）、max_iter:最大繰り返し数
        self.lambda_ = lambda_
        self.tol = tol
        self.max_iter = max_iter
        self.w_ = None

    def fit(self, X, t):
        n, d = X.shape
        self.w_ = np.zeros(d + 1)
        avgl1 = 0.
        for _ in range(self.max_iter):
            avgl1_prev = avgl1
            self._update(n, d, X, t)
            avgl1 = np.abs(self.w_).sum() / self.w_.shape[0] #wのL1ノムルを次元数で割った値(|w|/d)
            if abs(avgl1 - avgl1_prev) <= self.tol: #(|w|/d)がtolイカになるまで_updateを呼び出してwのすべての成分を更新する
                break

    def _update(self, n, d, X, t):
        self.w_[0] = (t - np.dot(X, self.w_[1:])).sum() / n
        w0vec = np.ones(n) * self.w_[0]
        for k in range(d):
            ww = self.w_[1:]
            ww[k] = 0
            q = np.dot(t - w0vec - np.dot(X, ww), X[:, k])
            r = np.dot(X[:, k], X[:, k])
            self.w_[k + 1] = soft_thresholding(q / r, self.lambda_)

    def predict(self, X):
        if X.ndim == 1:
            X = X.reshape(X.shape[0], 1)

        Xtil = np.c_[np.ones(X.shape[0]), X]
        return np.dot(Xtil, self.w_)

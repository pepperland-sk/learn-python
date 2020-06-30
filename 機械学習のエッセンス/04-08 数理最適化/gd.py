import numpy as np

class GradientDescent: #勾配配下法
    def __init__(self, f, df, alpha=0.01, eps=1e-6):
        self.f = f #最小化したい関数
        self.df = df #最小化したい導関数
        self.alpha = alpha  #α: 探索中の移動の大きさ
        self.eps = eps #eps: アルゴリズムの終了条件の基準を表す。∇fのL2ノルムがeps以下のときに終了
        self.path = None

    def solve(self, init):
        x = init
        path = []
        grad = self.df(x)
        path.append(x)
        while (grad**2).sum() > self.eps**2:
            x = x - self.alpha * grad
            grad = self.df(x)
            path.append(x)
        self.path_ = np.array(path) #解の記録
        self.x_ = x #計算結果としての最適解
        self.opt_ = self.f(x) #最適値

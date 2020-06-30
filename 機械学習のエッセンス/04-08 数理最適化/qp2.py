"""

■問
Minimize f(x, y) = x^2 + xy + y^2 +2x + 4y
Subject to x + y = 0

■2次計画問題
"""

import numpy as np
import cvxopt

P = cvxopt.matrix(np.array([[2,1], [1,2]], dtype=np.float64))
q = cvxopt.matrix(np.array([2,4], dtype=np.float64))
A = cvxopt.matrix(np.array([[1,1]], dtype=np.float64))
b = cvxopt.matrix(np.array([0], dtype=np.float64))

sol = cvxopt.solvers.qp(P, q, A=A, b=b) #Ax=bの制約式を書く

print(np.array(sol["x"])) #最適解を求める
print(np.array(sol["primal objective"])) #最適解の場合の目的関数の値を求める

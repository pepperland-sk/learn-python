"""


■問題：
製品X, Yを製造している。これらは原料A,B,Cから作られている。
利益を最大化するためのX, Yを製造すべき量を計算する。

(製品を作るのに必要な原材料)
X: A 1kg B 2kg C 2kg
Y: A 4kg B 3kg C 1kg

(原材料の量)
A: 1700kg
B: 1400kg
C: 1000kg

(製品を売るときの利益)
X: 3ドル
Y: 4ドル

■線形計画法：
Minimize cTx → 3x + 4y
Subject to Gx <= h, Ax = b
"""


import numpy as np
from scipy import optimize

c = np.array([-3, -4], dtype=np.float64) #c
G = np.array([[1, 4], [2, 3], [2, 1]], dtype=np.float64) #G
h = np.array([1700, 1400, 1000], np.float64) #h
sol = optimize.linprog(c, A_ub=G, b_ub=h, bounds=(0, None)) #bounds: 下限と上限を指定

print(sol.x)
print(sol.fun)

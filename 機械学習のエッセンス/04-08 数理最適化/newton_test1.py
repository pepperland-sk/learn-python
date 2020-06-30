"""
■問 以下の連立方程式の解をニュートン法によって求める
f1(x, y) = x^3 - 2y = 0
f2(x, y) = x^2 + y^2 - 1 = 0

■解法
初期値(1,1), (-1,-1), (1,-1)を使ってそれぞれ方程式を解く
"""

import numpy as np
import matplotlib.pyplot as plt
import newton

def f1(x, y):
    return x**3-2*y

def f2(x, y):
    return x**2+y**2-1

def f(xx):
    x = xx[0]
    y = xx[1]
    return np.array([f1(x, y), f2(x, y)])

def df(xx):
    x = xx[0]
    y = xx[1]
    return np.array([[3*x**2, -2], [2*x, 2*y]])

xmin, xmax, ymin, ymax = -3, 3, -3, 3
plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)
x = np.linspace(xmin, xmax, 200)
y = np.linspace(ymin, ymax, 200)
xmesh, ymesh = np.meshgrid(x, y)
z1 = f1(xmesh, ymesh) #f1を定義
z2 = f2(xmesh, ymesh) #f2を定義
plt.contour(xmesh, ymesh, z1, colors="r", levels=[0]) #f1(3次方程式の曲線)を描画
plt.contour(xmesh, ymesh, z2, colors="k", levels=[0]) #f2(楕円形)を描画
solver = newton.Newton(f, df)

initials = [np.array([1,1]), np.array([-1,-1]), np.array([1, -1])]
markers = ["+", "*", "x"]

for x0, m in zip(initials, markers):
    sol = solver.solve(x0) #初期値を与えてニュートン法を実行
    plt.scatter(solver.path_[:, 0], solver.path_[:, 1], color="k", marker=m) #xの軌跡を描画
    print(sol) #解の値を表示

plt.show()

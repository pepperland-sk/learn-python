def newton1dim(f, df, x0, eps=1e-10, max_iter=1000): #x0: 初期値 eps: 収束条件となるε max_iter: 繰り返しの最大回数
    x = x0
    iter = 0
    while True:
        x_new = x - f(x)/df(x)
        if abs(x-x_new) < eps:
            break
        x = x_new
        iter += 1
        if iter == max_iter:
            break
    return x_new

def f(x): #解を求めたい関数
    return x**3-5*x+1

def df(x): #導関数
    return 3*x**2-5

print(newton1dim(f, df, 2))
print(newton1dim(f, df, 0))
print(newton1dim(f, df, -3))

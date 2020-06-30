import numpy as np
import matplotlib.pyplot as plt

def cointoss(n, m): #n枚のコインを投げることをm回繰り返し、結果をリストで返す
    l = []
    for _ in range(m):
        r = np.random.randint(2, size = n)
        l.append(r.sum())
    return l

np.random.seed(0)
fig, axes = plt.subplots(1,2)

l1 = cointoss(100, 1000000) #100枚のコインを投げることを100000回繰り返す
axes[0].hist(l1, range(30, 70), bins=50, color="k")

l2 = cointoss(10000, 1000000)
axes[1].hist(l2, range(4800, 52000), bins=50, color="k")
plt.show()

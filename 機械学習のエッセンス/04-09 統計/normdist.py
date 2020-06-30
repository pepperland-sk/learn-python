import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

x = np.linspace(-5, 5)
y = norm.pdf(x) #norm.pdf(loc, scale): f(x) ~ N(loc,scale)となるf(x)の値を返す
plt.plot(x, y, color="r")
plt.show()

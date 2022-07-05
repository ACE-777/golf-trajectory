import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import curve_fit


def dv(v, t, a):
    return a * v * v


def v(t, a, v0):
    values = odeint(dv, v0, t, args=(a,))
    return values.ravel()


def x(t, a, v0):
    return v(t, a, v0) * t


data_t = np.array([0, 1, 2, 3, 4])
data_x = np.array([0, 1, 1.5, 1.8, 1.9])

vals, cov = curve_fit(x, data_t, data_x, [-2, 1])
a, v0 = vals
print('a: {} v0: {}'.format(a, v0))

t = np.linspace(0, 4, 20)

# By using the matplotlib.pyplot library we plot the curve after integration
plt.rcParams.update({'font.size': 14})  # increase the font size
plt.xlabel("t")
plt.ylabel("distance")
plt.title("x'' = a * (x')^2")
plt.plot(
    data_t, data_x, '.',
    t, x(t, a, v0), '-'
)
plt.show()

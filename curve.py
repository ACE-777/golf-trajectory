import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import curve_fit


def dvx(v, t, a):
    return a * v * v


def vx(t, a, v0):
    values = odeint(dvx, v0, t, args=(a,))
    return values.ravel()


def x(t, a, v0):
    return vx(t, a, v0) * t


def dvy(v, t, a, g):
    return a * v * v - g


def vy(t, a, g, v0):
    values = odeint(dvy, v0, t, args=(a, g))
    return values.ravel()


def y(t, a, g, v0):
    return vy(t, a, g, v0) * t


data_t = np.array([0, 1, 2, 3, 4])
data_x = np.array([0, 10, 15, 18, 19])
data_y = np.array([1, 5, 7, 7, 6])

vals_x, _ = curve_fit(x, data_t, data_x, [-2, 10])
vals_y, _ = curve_fit(y, data_t, data_y, [-2, -10, 10])
ax, v0x = vals_x
ay, g, v0y = vals_y
print('ax: {} v0: {} ay: {}, g: {}, v0y: {}'.format(ax, v0x, ay, g, v0y))

t = np.linspace(0, 4, 20)

# By using the matplotlib.pyplot library we plot the curve after integration
plt.rcParams.update({'font.size': 14})  # increase the font size
plt.xlabel("x")
plt.ylabel("y")
plt.title("x'' = ax * (x')^2, y'' = ay * (y')^2 - g")
plt.plot(
    data_x, data_y, '.',
    x(t, ax, v0x), y(t, ay, g, v0x), '-'
)
plt.show()

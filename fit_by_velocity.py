import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def poly(t, a, v0, start): return a * t ** 2 + v0 * t + start


velocity = 10
angle = math.pi / 4
xv0 = velocity * math.cos(angle)
yv0 = velocity * math.sin(angle)

def x1(t, xa): return poly(t, xa, 10 * math.cos(angle), 0)
def y1(t, ya): return poly(t, ya, 10 * math.sin(angle), 0)
def x2(t, xa): return poly(t, xa, 20 * math.cos(angle), 0)
def y2(t, ya): return poly(t, ya, 20 * math.sin(angle), 0)
def x3(t, xa): return poly(t, xa, 5 * math.cos(angle), 0)
def y3(t, ya): return poly(t, ya, 5 * math.sin(angle), 0)

(xa1), _ = curve_fit(x1, [0, 1], [0, 10])
(ya1), _ = curve_fit(y1, [0, 1], [0, 5])
(xa2), _ = curve_fit(x2, [0, 1], [0, 10])
(ya2), _ = curve_fit(y2, [0, 1], [0, 5])
(xa3), _ = curve_fit(x3, [0, 1], [0, 10])
(ya3), _ = curve_fit(y3, [0, 1], [0, 5])

times = np.linspace(0, 1.5, 20)
fig, axs = plt.subplots(1)
fig.suptitle('camera coords')
axs.set(xlabel='ksi', ylabel='eta')
axs.plot(
    x1(times, xa1), y1(times, ya1), '-',
    x2(times, xa2), y2(times, ya2), '-',
    x3(times, xa3), y3(times, ya3), '-',
)
plt.show()
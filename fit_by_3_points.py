import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def poly(t, a, v0, start): return a * t ** 2 + v0 * t + start


track1 = np.array([
    [323.80, 1093.22],
    [187.68, 640.46],
    [175.71, 626.36],
])

track2 = np.array([
    [571.50, 954.66],
    [530.44, 791.04],
    [509.41, 703.11],
])

track2_landing = np.array([
    [571.50, 954.66],
    [530.44, 791.04],
    [403.93, 474.01],
])

track1_i = np.arange(start=0, stop=len(track1), step=1)
track1_t = np.array(list(map(lambda i: i / 30, track1_i)))
track1_x = np.array(list(map(lambda v: v - 1080 / 2, track1[:, 0])))
track1_y = np.array(list(map(lambda v: 1920 / 2 - v, track1[:, 1])))

track2_i = np.arange(start=0, stop=len(track2), step=1)
track2_t = np.array(list(map(lambda i: i / 30, track2_i)))
track2_x = np.array(list(map(lambda v: v - 1080 / 2, track2[:, 0])))
track2_y = np.array(list(map(lambda v: 1920 / 2 - v, track2[:, 1])))

track2_landing_i = np.append(np.arange(start=0, stop=len(track2_landing) - 1, step=1), [48])
track2_landing_t = np.array(list(map(lambda i: i / 30, track2_landing_i)))
track2_landing_x = np.array(list(map(lambda p: p[0], track2_landing)))
track2_landing_y = np.array(list(map(lambda p: p[1], track2_landing)))

(xa1, xb1, xc1), _ = curve_fit(poly, track1_t, track1_x)
(ya1, yb1, yc1), _ = curve_fit(poly, track1_t, track1_y)
(xa2, xb2, xc2), _ = curve_fit(poly, track2_t, track2_x)
(ya2, yb2, yc2), _ = curve_fit(poly, track2_t, track2_y)

times = np.linspace(0, 5/30, 20)
fig, axs = plt.subplots(1)
fig.suptitle('camera coords')
axs.set(xlabel='ksi', ylabel='eta')
axs.plot(
    # poly(times, xa1, xb1, xc1), poly(times, ya1, yb1, yc1), '-',
    poly(times, xa2, xb2, xc2), poly(times, ya2, yb2, yc2), '-',
)
plt.show()
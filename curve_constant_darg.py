import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint, quad
from scipy.optimize import curve_fit


# z - from the camera to the far
# x - horizontal
# y - vertical

def x(t, drag, v0, x0): return drag * t**2 / 2 + v0 * t + x0
def z(t, drag, v0, z0): return drag * t**2 / 2 + v0 * t + z0
def y(t, drag, v0, y0, g): return (drag + g) * t**2 / 2 + v0 * t + y0


def ksi(t, xa, xv0, x0, za, zv0, z0): return focal * x(t, xa, xv0, x0) / z(t, za, zv0, z0)
def eta(t, ya, yv0, y0, za, zv0, z0, g): return focal * y(t, ya, yv0, y0, g) / z(t, za, zv0, z0)


def fit_3d():
    data_t = np.array([0, 0.3, 0.6, 0.9, 1.2])
    data_z = np.array([0, 10, 15, 18, 19])
    data_y = np.array([1, 5, 7, 7, 6])

    vals_z, _ = curve_fit(x, data_t, data_z)
    g = -9.8
    def y_fixed(t, drag, v0, y0): return y(t, drag, v0, y0, g)
    vals_y, _ = curve_fit(y_fixed, data_t, data_y)

    za, zv0, z0 = vals_z
    ya, yv0, y0 = vals_y
    print('zv0 {} z0 {} yv0 {} y0 {} g {}'.format(zv0, z0, yv0, y0, g))

    t = np.linspace(0, 2, 20)

    fig, axs = plt.subplots(3)
    fig.suptitle('3d coordinates')

    axs[0].set(xlabel='z', ylabel='y')
    axs[0].plot(
        data_z, data_y, '.',
        z(t, za, zv0, z0), y(t, ya, yv0, y0, g), '-'
    )

    axs[1].set(xlabel='t', ylabel='y')
    axs[1].plot(
        t,  y(t, ya, yv0, y0, g), '-'
    )

    axs[2].set(xlabel='t', ylabel='z')
    axs[2].plot(
        t, z(t, za, zv0, z0), '-'
    )
    plt.show()


def fit_2d():
    track = np.array([
        [323.80, 1093.22],
        [187.68, 640.46],
        [175.71, 626.36],
        [174.96, 655.97],
        [179.48, 755.04],
        [184.92, 829.45]
    ])

    track_ksi = np.array(list(map(lambda v: v - 1080 / 2, track[:, 0])))
    track_eta = np.array(list(map(lambda v: 1920 / 2 - v, track[:, 1])))
    track_t = np.array([0, 0.33, 0.66, 0.99, 1.3, 1.6])

    g = -9.8

    vals_x, _ = curve_fit(ksi, track_t, track_ksi)
    xa, xv0, x0, za, zv0, z0 = vals_x

    def eta_fixed(t, ya, yv0, y0):
        return eta(t, ya, yv0, y0, za, zv0, z0, g)

    vals_y, _ = curve_fit(eta_fixed, track_t, track_eta)
    ya, yv0, y0 = vals_y
    print('xa {}, za {}, xv0 {}, zv0 {}, x0 {}, z0 {}'.format(xa, za, xv0, zv0, x0, z0))
    print('ya {}, yv0 {}, y0 {}, g {}'.format(ya, yv0, y0, g))

    t = np.linspace(0, 2, 20)

    fig, axs = plt.subplots(3)
    fig.suptitle('3d and camera projection')

    axs[0].set(xlabel='ksi', ylabel='eta')
    axs[0].plot(
        track_ksi, track_eta, '.',
        ksi(t, xa, za, xv0, zv0, x0, z0), eta(t, ya, yv0, y0, za, zv0, z0, g), '-'
    )

    axs[1].set(xlabel='t', ylabel='y')
    axs[1].plot(
        t, y(t, ya, g, yv0, y0), '-'
    )

    axs[2].set(xlabel='t', ylabel='z')
    axs[2].plot(
        t, z(t, za, zv0, z0), '-'
    )

    plt.show()


focal = 0.013
fit_3d()
# fit_2d()

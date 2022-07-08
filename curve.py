import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import curve_fit

# z - from the camera to the far
# x - horizontal
# y - vertical

def dvx(v, t, a): return a * v * v
def vx(t, a, v0): return odeint(dvx, v0, t, args=(a,)).ravel()
def x(t, a, v0, x0): return vx(t, a, v0) * t + x0

def vz(t, a, v0): return odeint(dvx, v0, t, args=(a,)).ravel()
def z(t, a, v0, z0): return vx(t, a, v0) * t + z0

def dvy(v, t, a, g): return a * v * v + g
def vy(t, a, g, v0): return odeint(dvy, v0, t, args=(a, g)).ravel()
def y(t, a, g, v0, y0): return vy(t, a, g, v0) * t + y0


# def d_ksi(t, xa, xv0, x0, za, zv0, z0):
#     z_value = z(t, za, zv0, z0)
#     z_val = np.array(z_value)
#     z_value[z_value == 0] = 0.0000001
#     return focal * (vx(t, xa, xv0) * z_val - x(t, xa, xv0, x0) * vz(t, za, zv0)) / z_value ** 2
#
#
# def d_eta(t, ya, za, yv0, zv0, y0, z0, g):
#     z_value = z(t, za, zv0, z0)
#     z_value[z_value == 0] = 0.0000001
#     return focal * (vy(t, ya, g, yv0) * z_value - y(t, ya, g, yv0, y0) * vz(t, za, zv0)) / z_value ** 2


def ksi(t, xa, xv0, x0, za, zv0, z0): return focal * x(t, xa, xv0, x0) / z(t, za, zv0, z0)
def eta(t, ya, yv0, y0, za, zv0, z0, g): return focal * y(t, ya, yv0, y0, g) / z(t, za, zv0, z0)


def fit_3d():
    data_t = np.array([0, 1, 2, 3, 4])
    data_z = np.array([0, 10, 15, 18, 19])
    data_y = np.array([1, 5, 7, 7, 6])
    vals_z, _ = curve_fit(x, data_t, data_z, [-0.2, 10, 0])
    vals_y, _ = curve_fit(y, data_t, data_y, [-0.2, -10, 10, 1])
    za, zv0, z0 = vals_z
    ay, g, v0y, y0 = vals_y
    print('za: {} zv0: {} ay: {}, g: {}, v0y: {}'.format(za, zv0, ay, g, v0y))

    t = np.linspace(0, 4, 20)

    plt.xlabel("z")
    plt.ylabel("y")
    plt.title("z'' = az * (z')^2, y'' = ay * (y')^2 - g")
    plt.plot(
        data_z, data_y, '.',
        x(t, za, zv0, z0), y(t, ay, g, v0y, y0), '-'
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

    z0 = 5
    zv0 = 10
    g = -9.8

    def ksi_fixed(t, xa, xv0, x0, za):
        return ksi(t, xa, xv0, x0, za, zv0, z0)

    vals_x, _ = curve_fit(ksi_fixed, track_t, track_ksi, [-1, 1, 0, -0.2])
    xa, xv0, x0, za = vals_x

    def eta_fixed(t, ya, yv0, y0):
        return eta(t, ya, yv0, y0, za, zv0, z0, g)

    vals_y, _ = curve_fit(eta_fixed, track_t, track_eta, [-1, 10, 1])
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
# fit_3d()
fit_2d()

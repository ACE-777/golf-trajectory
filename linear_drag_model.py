import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from math import exp, log, cosh, sqrt

focal = 0.013
pixel_to_meter = 1920 / 0.03

# z - from the camera to the far
# x - horizontal
# y - vertical

m = 0.0459
z0 = 1
g = 9.8


def horizontal_drag_coord(t, k, v0, x0):
    kt_exp = np.array([exp(ti * k) for ti in t])
    return -v0 / k + v0 / k * kt_exp + x0


x = horizontal_drag_coord
z = horizontal_drag_coord


def y(t, k, v0, y0):
    kt_exp = np.array([exp(ti * k) for ti in t])
    return 1 / k * (v0 - g / k) * kt_exp + g / k * t - 1 / k * (v0 - g / k) + y0


def ksi(t, k, xv0, x0, vz0): return focal * x(t, k, xv0, x0) / z(t, k, vz0, z0) * pixel_to_meter


def eta(t, k, vy0, vz0, y0): return focal * y(t, k, vy0, y0) / z(t, k, vz0, z0) * pixel_to_meter


def fit_linear_drag(track, track_t, extrapolate_to=None):
    if extrapolate_to is None:
        extrapolate_to = track_t[-1] * 3
    t = np.linspace(0, extrapolate_to, 20)

    track_ksi = np.array(list(map(lambda v: v - 1080 / 2, track[:, 0])))
    track_eta = np.array(list(map(lambda v: 1920 / 2 - v, track[:, 1])))

    bounds = ((-10, 0, 1, -20),
              (1, 200, 100, 20))
    vals_eta, _ = curve_fit(eta, track_t, track_ksi, bounds=bounds)
    k, vy0, vz0, y0 = vals_eta

    print('k {:.2f}, vy0 {:.2f}, vz0 {:.2f}, y0 {:.2f}'.format(k, vy0, vz0, y0))

    def ksi_fixed(t, xv0, x0):
        return ksi(t, k, xv0, x0, vz0)

    bounds = ((-20, -2),
              (20, 2))
    vals_ksi, _ = curve_fit(ksi_fixed, track_t, track_eta, bounds=bounds)
    xv0, x0 = vals_ksi
    print('xv0 {:.2f}, x0 {:.2f}'.format(xv0, x0))

    fig, axs = plt.subplots(3)
    fig.suptitle('3d and camera projection')

    axs[0].set(xlabel='ksi', ylabel='eta')
    axs[0].plot(
        track_ksi, track_eta, '.',
        ksi(t, k, xv0, x0, vz0), eta(t, k, vy0, vz0, y0), '-'
    )

    axs[1].set(xlabel='t', ylabel='y')
    axs[1].plot(
        t, y(t, k, vy0, y0), '-'
    )

    axs[2].set(xlabel='t', ylabel='z')
    axs[2].plot(
        t, z(t, k, vz0, z0), '-'
    )

    plt.show()


def fit_linear_drag_3d():
    data_t = np.array([0, 0.3, 0.6, 0.9, 1.2])
    data_z = np.array([0, 10, 15, 18, 19])
    data_y = np.array([1, 5, 7, 7, 6])
    vals_z, _ = curve_fit(z, data_t, data_z)
    k, vz0, z0 = vals_z

    def y_fixed(t, v0, y0):
        return y(t, k, v0, y0)

    vals_y, _ = curve_fit(y_fixed, data_t, data_y)
    vy0, y0 = vals_y

    t = np.linspace(0, 2, 20)

    fig, axs = plt.subplots(3)
    fig.suptitle('3d coordinates')

    axs[0].set(xlabel='z', ylabel='y')
    axs[0].plot(
        data_z, data_y, '.',
        z(t, k, vz0, z0), y(t, k, vy0, y0), '-'
    )

    axs[1].set(xlabel='t', ylabel='y')
    axs[1].plot(
        t,  y(t, k, vy0, y0), '-'
    )

    axs[2].set(xlabel='t', ylabel='z')
    axs[2].plot(
        t, z(t, k, vz0, z0), '-'
    )
    plt.show()


if __name__ == '__main__':
    fit_linear_drag_3d()

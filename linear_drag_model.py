import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from math import exp, log, cosh, sqrt

from curve import prepare_times

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


def fit_linear_drag(track, target_times):
    if np.shape(track)[1] == 3:
        track_t = track[:, 2]
    else:
        track_t = np.arange(0, (len(track)) / 30, 1/30)

    track_ksi = track[:, 0]
    track_eta = track[:, 1]

    bounds = ((-3, 1, 1, -10),
              (5, 10, 100, 10))
    vals_eta, _ = curve_fit(eta, track_t, track_eta, bounds=bounds)
    k, vy0, vz0, y0 = vals_eta

    print('k {:.2f}, vy0 {:.2f}, vz0 {:.2f}, y0 {:.2f}'.format(k, vy0, vz0, y0))

    def ksi_fixed(t, xv0, x0):
        return ksi(t, k, xv0, x0, vz0)

    bounds = ((-20, -2),
              (20, 2))
    vals_ksi, _ = curve_fit(ksi_fixed, track_t, track_ksi, bounds=bounds)
    xv0, x0 = vals_ksi
    print('xv0 {:.2f}, x0 {:.2f}'.format(xv0, x0))

    def eta_by_t(t):
        return eta(t, k, vy0, vz0, y0)

    t = prepare_times(target_times, eta_by_t, target_times)
    xs = ksi(t, k, xv0, x0, vz0)
    ys = eta(t, k, vy0, vz0, y0)
    return np.stack((xs, ys, t), axis=1)


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

import math

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
y0 = 0


def horizontal_drag_coord(t, c_drag, v0, val0):
    if v0 == 0 or c_drag == 0:
        return val0
    tau = m / c_drag * v0
    temp = (1 + t / tau)
    temp[temp < 0] = 1
    ln = np.array([log(val) for val in temp])
    return v0 * tau * ln + val0


x = horizontal_drag_coord
z = horizontal_drag_coord


def y(t, a, b): return a * m * t ** 2 + b * t + y0


def ksi(t, c_drag_x, c_drag_z, xv0, x0, zv0, z0): return focal * x(t, c_drag_x, xv0, x0) / z(t, c_drag_z, zv0,
                                                                                             z0) * pixel_to_meter


def eta(t, a, b, c_drag_z, zv0, z0): return focal * y(t, a, b) / z(t, c_drag_z, zv0, z0) * pixel_to_meter


def ksi_simple(t, a, b, ksi0): return a * t ** 2 + b * t + ksi0


def eta_simple(t, a, b, eta0): return a * t ** 2 + b * t + eta0


def fit_3d():
    data_t = np.array([0, 0.3, 0.6, 0.9, 1.2])
    data_z = np.array([0, 10, 15, 18, 19])
    data_y = np.array([1, 5, 7, 7, 6])
    vals_z, _ = curve_fit(z, data_t, data_z)
    c_drag, zv0, z0 = vals_z

    def y_fixed(t, a, b): return y(t, a, b)

    vals_y, _ = curve_fit(y_fixed, data_t, data_y)
    a, b = vals_y

    print(vals_z)
    print(z(data_t, c_drag, zv0, z0))

    t = np.linspace(0, 2, 20)

    fig, axs = plt.subplots(3)
    fig.suptitle('3d coordinates')

    axs[0].set(xlabel='z', ylabel='y')
    axs[0].plot(
        data_z, data_y, '.',
        z(t, c_drag, zv0, z0), y(t, a, b), '-'
    )

    axs[1].set(xlabel='t', ylabel='y')
    axs[1].plot(
        t, y(t, a, b), '-'
    )

    axs[2].set(xlabel='t', ylabel='z')
    axs[2].plot(
        t, z(t, c_drag, zv0, z0), '-'
    )
    plt.show()


def fit_2d(track, track_t, extrapolate_to=None):
    if extrapolate_to is None:
        extrapolate_to = track_t[-1] * 3
    t = np.linspace(0, extrapolate_to, 20)

    track_ksi = np.array(list(map(lambda v: v - 1080 / 2, track[:, 0])))
    track_eta = np.array(list(map(lambda v: 1920 / 2 - v, track[:, 1])))

    ksi_param_bounds = ((-3, -0.001, -25, -25, 20, 1),
                        (3, 0, 20, 20, 40, 10))
    vals_ksi, _ = curve_fit(ksi, track_t, track_ksi, bounds=ksi_param_bounds, method='trf')
    c_drag_x, c_drag_z, xv0, x0, zv0, z0 = vals_ksi

    print(
        'c_drag_x {:.2f}, c_drag_z {:.4f} xv0 {:.2f}, x0 {:.2f}, zv0 {:.2f}, z0 {:.2f}'.format(c_drag_x, c_drag_z, xv0,
                                                                                               x0, zv0, z0))

    def eta_fixed(t, a, b):
        return eta(t, a, b, c_drag_z, zv0, z0)

    eta_param_bounds = ((-4000, 10),
                        (-50, 40))
    vals_eta, _ = curve_fit(eta_fixed, track_t, track_eta, bounds=eta_param_bounds, method='trf')
    a, b = vals_eta
    print('y_a {:.2f}, y_b {:.2f}'.format(a, b))

    fig, axs = plt.subplots(3)
    fig.suptitle('3d and camera projection')

    axs[0].set(xlabel='ksi', ylabel='eta')
    axs[0].plot(
        track_ksi, track_eta, '.',
        ksi(t, c_drag_x, c_drag_z, xv0, x0, zv0, z0), eta(t, a, b, c_drag_z, zv0, z0), '-'
    )

    axs[1].set(xlabel='t', ylabel='y')
    axs[1].plot(
        t, y(t, a, b), '-'
    )

    axs[2].set(xlabel='t', ylabel='z')
    axs[2].plot(
        t, z(t, c_drag_z, zv0, z0), '-'
    )

    plt.show()


def fit_quadratic_drag(track, target_times=None):
    time_track = np.array(track)
    track_t = time_track[:, 2]

    track_ksi = np.array(list(map(lambda v: v - 1080 / 2, time_track[:, 0])))
    track_eta = np.array(list(map(lambda v: 1920 / 2 - v, time_track[:, 1])))

    ksi_param_bounds = ((-3, -0.001, -25, -25, 20, 1),
                        (3, 0, 20, 20, 40, 10))
    vals_ksi, _ = curve_fit(ksi, track_t, track_ksi, bounds=ksi_param_bounds, method='trf')
    c_x, c_z, xv0, x0, zv0, z0 = vals_ksi

    print('c_x {:.2f}, c_z {:.4f} xv0 {:.2f}, x0 {:.2f}, zv0 {:.2f}, z0 {:.2f}'.format(c_x, c_z, xv0, x0, zv0, z0))

    def eta_fixed(t, a, b):
        return eta(t, a, b, c_z, zv0, z0)

    eta_param_bounds = ((-4000, 10),
                        (-50, 40))
    vals_eta, _ = curve_fit(eta_fixed, track_t, track_eta, bounds=eta_param_bounds, method='trf')
    a, b = vals_eta
    print('y_a {:.2f}, y_b {:.2f}'.format(a, b))

    if target_times is None:
        start = track_t[0]
        target_times = np.linspace(start,  start + 10, 100)
        t = np.array(target_times)
        ys = eta(t, a, b, c_z, zv0, z0)
        max_y = np.argmax(ys)
        end = start + (max_y - start) * 2
        step = track_t[1] - start
        frames = math.ceil((end - start) / step)
        target_times = np.linspace(start, start + frames * step, frames)

    t = np.array(target_times)
    xs = ksi(t, c_x, c_z, xv0, x0, zv0, z0)
    ys = eta(t, a, b, c_z, zv0, z0)
    xs = np.array(list(map(lambda v: v + 1080 / 2, xs)))
    ys = np.array(list(map(lambda v: 1920 / 2 - v, ys)))
    return np.stack((xs, ys, t), axis=1)

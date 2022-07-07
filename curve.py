import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import curve_fit


def dvx(v, t, a):
    return a * v * v


def vx(t, a, v0):
    values = odeint(dvx, v0, t, args=(a,))
    return values.ravel()


def vz(t, a, v0):
    return odeint(dvx, v0, t, args=(a,)).ravel()


def x(t, a, v0, x0):
    return vx(t, a, v0) * t + x0


def z(t, a, v0, z0):
    return vx(t, a, v0) * t + z0


def dvy(v, t, a, g):
    return a * v * v - g


def vy(t, a, g, v0):
    values = odeint(dvy, v0, t, args=(a, g))
    return values.ravel()


def y(t, a, g, v0, y0):
    return vy(t, a, g, v0) * t + y0


def ksi(t, xa, za, xv0, zv0, x0, z0):
    z_value = z(t, za, zv0, z0)
    z_val = np.array(z_value)
    z_value[z_value == 0] = 0.0000001
    return focal * (vx(t, xa, xv0) * z_val - x(t, xa, xv0, x0) * vz(t, za, zv0)) / z_value ** 2


def eta(t, ya, za, yv0, zv0, y0, z0, g):
    z_value = z(t, za, zv0, z0)
    z_value[z_value == 0] = 0.0000001
    return focal * (vy(t, ya, g, yv0) * z_value - y(t, ya, g, yv0, y0) * vz(t, za, zv0)) / z_value ** 2


def fit_3d():
    data_t = np.array([0, 1, 2, 3, 4])
    data_x = np.array([0, 10, 15, 18, 19])
    data_y = np.array([1, 5, 7, 7, 6])
    vals_x, _ = curve_fit(x, data_t, data_x, [-0.2, 10, 0])
    vals_y, _ = curve_fit(y, data_t, data_y, [-0.2, -10, 10, 1])
    ax, v0x, x0 = vals_x
    ay, g, v0y, y0 = vals_y
    print('ax: {} v0: {} ay: {}, g: {}, v0y: {}'.format(ax, v0x, ay, g, v0y))

    t = np.linspace(0, 4, 20)

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("x'' = ax * (x')^2, y'' = ay * (y')^2 - g")
    plt.plot(
        data_x, data_y, '.',
        x(t, ax, v0x, x0), y(t, ay, g, v0y, y0), '-'
    )
    plt.show()


def fit_2d():
    track = np.array([
        [323.80, 1093.22],
        [274.14, 922.17],
        [246.67, 828.60],
        [229.30, 770.31],
        [217.66, 731.01],
        [209.00, 703.70],
        [202.46, 683.17],
        [197.86, 667.91],
        [193.69, 656.75],
        [190.59, 647.53],
        [187.68, 640.46],
        [185.62, 635.00],
        [183.74, 631.35],
        [181.82, 627.53],
        [180.70, 625.85],
        [179.69, 624.61],
        [178.72, 624.30],
        [177.73, 623.75],
        [176.76, 624.61],
        [176.71, 625.60],
        [175.71, 626.36],
        [175.87, 628.42],
        [175.21, 630.60],
        [174.94, 633.30],
        [174.83, 635.32],
        [174.87, 638.74],
        [174.74, 641.57],
        [174.61, 644.45],
        [174.73, 648.29],
        [174.85, 652.13],
        [174.96, 655.97],
        [175.34, 659.41],
        [175.45, 663.97],
        [175.56, 668.54]
    ])

    track_ksi = track[:, 0]
    track_eta = np.array(list(map(lambda y: 2000 - y, track[:, 1])))
    track_t = np.array(range(len(track)))

    # z0 = 1
    # def ksi_fixed(t, xa, za, xv0, zv0, x0):
    #     return ksi(t, xa, za, xv0, zv0, x0, z0)

    vals_x, _ = curve_fit(ksi, track_t, track_ksi, [-0.2, -0.2, 10, 10, 0, 1])
    xa, za, xv0, zv0, x0, z0 = vals_x

    def eta_fixed(t, ya, yv0, y0, g):
        return eta(t, ya, za, yv0, zv0, y0, z0, g)

    vals_y, _ = curve_fit(eta_fixed, track_t, track_eta, [-0.2, 10, 0, -10])
    ya, yv0, y0, g = vals_y
    print('xa {}, za {}, xv0 {}, zv0 {}, x0 {}, z0 {}'.format(xa, za, xv0, zv0, x0, z0))

    t = np.linspace(0, len(track), 20)

    plt.xlabel("ksi")
    plt.ylabel("eta")
    plt.title("x'' = ax * (x')^2, y'' = ay * (y')^2 - g")
    plt.plot(
        track_ksi, track_eta, '.',
        ksi(t, xa, za, xv0, zv0, x0, z0), eta(t, ya, za, yv0, zv0, y0, z0, g), '-'
    )
    plt.show()


focal = 0.13
# fit_3d()
fit_2d()

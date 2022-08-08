import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit, minimize
from curve import focal, pixel_to_meter, prepare_times
from scipy.spatial.distance import cdist

meter_to_feet = 3.28084


# https://www.math.union.edu/~wangj/courses/previous/math238w13/Golf%20Ball%20Flight%20Dynamics2.pdf
def magnus_derivatives(t, x, w_i, w_k):
    c_d = .15
    r = .002378
    a = 0.25 * math.pi * (1.75 / 12) * (1.75 / 12)
    # D =((1/2)*C_d*r*A*V^2
    d = ((1 / 2) * c_d * r * a)
    # Magnus
    s = .000005
    m = (1.5 / (16 * 32.2))
    magnus = (s / m)
    w_j = 0

    x_prime = np.zeros(6)
    # X
    x_prime[0] = x[1]
    x_prime[1] = -(d / m) * x[1] ** 2 + magnus * (w_j * x[5] - w_k * x[3])
    # Y
    x_prime[2] = x[3]
    x_prime[3] = -32.2 - (d / m) * x[3] ** 2 + magnus * (w_k * x[1] - w_i * x[5])
    # Z
    x_prime[4] = x[5]
    x_prime[5] = -(d / m) * x[5] ** 2 + magnus * (w_i * x[3] - w_j * x[1])
    return x_prime


def magnus_derivatives_fixed_spin(t, x):
    return magnus_derivatives(t, x, -100, 100)


def visualize_magnus(t_eval):
    t_span = (t_eval[0], t_eval[-1])
    y0 = [0, 175, 0, 75, 0, 0]
    results = solve_ivp(magnus_derivatives_fixed_spin, t_span, y0, t_eval=t_eval)
    y = results.y

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    fig.suptitle('Golf ball Magnus model')

    print('x: {}'.format(y[0, :]))
    print('y: {}'.format(y[2, :]))
    print('z: {}'.format(y[4, :]))

    ax.plot3D(y[0, :], y[4, :], y[2, :], 'gray')
    plt.show()


def magnus_coord(t, w_i, w_k, v_x, v_y, v_z):
    t_span = (t[0], t[-1])
    y0 = [0, v_x, 0, v_y, 0, v_z]

    def derivatives(t, x):
        return magnus_derivatives(t, x, w_i, w_k)

    return solve_ivp(derivatives, t_span, y0, t_eval=t).y


def ksi(t, w_i, w_k, v_x, v_y, v_z, z0, x0):
    coords = magnus_coord(t, w_i, w_k, v_x, v_y, v_z)
    x = coords[4, :] + x0
    z = coords[0, :] + z0
    return focal * x / z * pixel_to_meter * meter_to_feet


def eta(t, w_i, w_k, v_x, v_y, v_z, z0, y0):
    coords = magnus_coord(t, w_i, w_k, v_x, v_y, v_z)
    y = coords[2, :] + y0
    z = coords[0, :] + z0
    return focal * y / z * pixel_to_meter * meter_to_feet


def fit_magnus(track, target_times):
    if np.shape(track)[1] == 3:
        track_t = track[:, 2]
    else:
        track_t = np.arange(0, (len(track)) / 30, 1 / 30)

    bounds = ((-100, -100, -20, 50, 20, 1, 0),
              (100, 100, 20, 200, 100, 10, 1))
    vals_eta, _ = curve_fit(eta, track_t, track[:, 1], method='trf', bounds=bounds)
    w_i, w_k, v_x, v_y, v_z, z0, y0 = vals_eta

    for v in vals_eta:
        print('v {:.2f}'.format(v))

    def ksi_fixed(t, x0):
        return ksi(t, w_i, w_k, v_x, v_y, v_z, z0, x0)

    vals_ksi, _ = curve_fit(ksi_fixed, track_t, track[:, 0], method='trf')
    x0 = vals_ksi[0]
    print('x0 {:.2f}'.format(x0))

    def eta_by_t(t):
        return eta(t, w_i, w_k, v_x, v_y, v_z, z0, y0)

    t = prepare_times(target_times, eta_by_t, target_times)
    xs = ksi(t, w_i, w_k, v_x, v_y, v_z, z0, x0)
    ys = eta(t, w_i, w_k, v_x, v_y, v_z, z0, y0)
    return np.stack((xs, ys, t), axis=1)


def distance_magnus(params, track):
    w_i, w_k, v_x, v_y, v_z, x0, y0, z0 = params
    t = track[:, 2]
    coords = magnus_coord(t, w_i, w_k, v_x, v_y, v_z)
    x = coords[4, :] + x0
    y = coords[2, :] + y0
    z = coords[0, :] + z0
    ksi = focal * x / z * pixel_to_meter * meter_to_feet
    eta = focal * y / z * pixel_to_meter * meter_to_feet
    points = np.stack((ksi, eta), axis=1)
    dists = cdist(points, track[:, [0, 1]])
    return sum(np.diagonal(dists)) / len(dists)


def minimize_magnus(track, target_times):
    bounds = ((-100, 100), (-100, 100), (50, 300), (1, 100), (-20, 20), (-1, 1), (0, 1), (3, 10))

    result = minimize(distance_magnus, x0=(0, 0, 0, 50, 100, 0, 0, 1), args=track, bounds=bounds)
    print(result)
    w_i, w_k, v_x, v_y, v_z, x0, y0, z0 = result.x

    def eta_by_t(t):
        return eta(t, w_i, w_k, v_x, v_y, v_z, z0, y0)

    t = prepare_times(target_times, eta_by_t, target_times)
    xs = ksi(t, w_i, w_k, v_x, v_y, v_z, z0, x0)
    ys = eta(t, w_i, w_k, v_x, v_y, v_z, z0, y0)
    return np.stack((xs, ys, t), axis=1)


if __name__ == '__main__':
    times = np.linspace(0, 10, 20)
    visualize_magnus(times)

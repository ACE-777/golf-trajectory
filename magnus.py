import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


# https://www.math.union.edu/~wangj/courses/previous/math238w13/Golf%20Ball%20Flight%20Dynamics2.pdf
def magnus_derivatives(t, x):
    c_d = .15
    r = .002378
    a = 0.25 * math.pi * (1.75 / 12) * (1.75 / 12)
    # D =((1/2)*C_d*r*A*V^2
    d = ((1 / 2) * c_d * r * a)
    # Magnus
    s = .000005
    m = (1.5 / (16 * 32.2))
    magnus = (s / m)
    w_i = -100
    w_j = 0
    w_k = 110

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


def visualize_magnus():
    t_span = (0, 8.3)
    y0 = [0, 175, 0, 75, 0, 0]
    results = solve_ivp(magnus_derivatives, t_span, y0)
    y = results.y

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    fig.suptitle('Golf ball Magnus model')

    print('x: {}'.format(y[0, :]))
    print('y: {}'.format(y[2, :]))
    print('z: {}'.format(y[4, :]))

    ax.plot3D(y[0, :], y[4, :], y[2, :], 'gray')
    plt.show()


if __name__ == '__main__':
    visualize_magnus()
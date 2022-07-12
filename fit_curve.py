import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from math import exp, log, cosh, sqrt
from curve import fit_2d

track = np.array([
    [323.80, 1093.22],
    [187.68, 640.46],
    [175.71, 626.36],
    [174.96, 655.97],
    [179.48, 755.04],
    [184.92, 829.45],
    [188.68, 902.82]
])

# test_ksi = focal * 10 / 20 * pixel_to_meter
# print(test_ksi)
# fit_3d()
fit_2d(track)
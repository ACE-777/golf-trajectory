import numpy as np
from linear_drag_model import fit_linear_drag

track = np.array([
    [323.80, 1093.22],
    [187.68, 640.46],
    [175.71, 626.36],
    [174.96, 655.97],
    [179.48, 755.04],
    [184.92, 829.45],
    [188.68, 902.82]
])

track2 = np.array([
    [571.50, 954.66],
    [530.44, 791.04],
    [509.41, 703.11],
    [494.57, 647.75],
    [483.32, 611.30],
    [477.30, 583.73],
    [471.29, 563.30],
])

track2_landing = np.array([
    [571.50, 954.66],
    [530.44, 791.04],
    [509.41, 703.11],
    [494.57, 647.75],
    [483.32, 611.30],
    [477.30, 583.73],
    [403.93, 474.01],
])

track1_i = np.arange(start=0, stop=len(track), step=1)
track1_t = np.array(list(map(lambda i: i / 30, track1_i)))

track2_i = np.arange(start=0, stop=len(track2), step=1)
track2_t = np.array(list(map(lambda i: i / 30, track2_i)))

track2_landing_i = np.append(np.arange(start=0, stop=len(track2_landing) - 1, step=1), [48])
track2_landing_t = np.array(list(map(lambda i: i / 30, track2_landing_i)))

fit_linear_drag(track, track1_t)
fit_linear_drag(track2_landing, track2_landing_t, 50/30)
fit_linear_drag(track2, track2_t)

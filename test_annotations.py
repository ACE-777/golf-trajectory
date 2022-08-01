import numpy as np
import sys
import os
import scipy
import matplotlib.pyplot as plt
from curve import fit_quadratic_drag
from cvat_annotations import load_track


def test_dataset(root, visualize):
    total_dist = 0
    total_tasks = 0
    for task in sorted(os.listdir(root)):
        if visualize and total_tasks > 3:
            break
        task_path = os.path.join(root, task)
        if not os.path.isdir(task_path):
            continue
        ann_path = os.path.join(task_path, 'annotations.json')
        track = load_track(ann_path)
        track_times = np.arange(0, (len(track)) / 30, 1 / 30)
        try:
            print("---- {} ----".format(task))
            result = fit_quadratic_drag(track[0:8], track_times)
            dists = scipy.spatial.distance.cdist(result[:, [0, 1]], track)
            dist = sum(np.diagonal(dists)) / len(dists)
            total_dist += dist
            total_tasks += 1
            print(dist)
            if visualize:
                plot(result, track)
        except RuntimeError:
            print("Could not fit the curve for {}".format(task_path))

    print("total distance: {}".format(total_dist / total_tasks))


def plot(result, track):
    track = np.array(track)
    fig, axs = plt.subplots(1)
    fig.suptitle('3d and camera projection')

    axs.set(xlabel='ksi', ylabel='eta')
    axs.plot(
        track[:, 0], track[:, 1], '.',
        result[:, 0], result[:, 1], '-'
    )
    plt.show()


if __name__ == '__main__':
    test_dataset(sys.argv[1], False)

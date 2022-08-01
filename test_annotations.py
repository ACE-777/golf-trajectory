import numpy as np
import sys
import os
import scipy

from curve import fit_quadratic_drag
from cvat_annotations import load_track


def test_dataset(root):
    total_dist = 0
    total_tasks = 0
    for task in sorted(os.listdir(root)):
        task_path = os.path.join(root, task)
        if not os.path.isdir(task_path):
            continue
        ann_path = os.path.join(task_path, 'annotations.json')
        track = load_track(ann_path)
        track_times = np.arange(0, (len(track)) / 30, 1/30)
        try:
            print("---- {} ----".format(task))
            result = fit_quadratic_drag(track[0:2], track_times)
            dists = scipy.spatial.distance.cdist(result[:, [0, 1]], track)
            dist = sum(np.diagonal(dists)) / len(dists)
            total_dist += dist
            total_tasks += 1
            print(dist)
        except RuntimeError:
            print("Could not fit the curve for {}".format(task_path))

    print("total distance: {}".format(total_dist / total_tasks))


if __name__ == '__main__':
    test_dataset(sys.argv[1])

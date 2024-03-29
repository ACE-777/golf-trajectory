from enum import Enum
from statistics import mean

import numpy as np
import sys
import os
import scipy
import matplotlib.pyplot as plt
from curve import fit_quadratic_drag, normalize_coordinates
from cvat_annotations import load_track
import av
import time


from linear_drag_model import fit_linear_drag
from magnus import minimize_magnus

white_list = []
skip_list = ['task_34']
points = 10
method = 'magnus'  # quadratic magnus


class FittingMode(Enum):
    Normal = 1
    LastPoint = 2
    ApexPoint = 3
    ApexAndLast = 4


def test_dataset(root, mode=FittingMode.Normal, visualize=False, algorithm='SLSQP', max_tasks=100):
    print('Starting test mode: {} algorithm: {} max_tasks: {}'.format(mode, algorithm, max_tasks))
    total_dists = []
    total_tasks = 0
    for task in sorted(os.listdir(root)):
        if len(white_list) > 0 and task not in white_list:
            continue
        if task in skip_list:
            continue
        if total_tasks > max_tasks:
            break
        task_path = os.path.join(root, task)
        if not os.path.isdir(task_path):
            continue
        print("---- {} ----".format(task))
        im_size = get_image_size(task_path)
        ann_path = os.path.join(task_path, 'annotations.json')
        track = load_track(ann_path)
        track = normalize_coordinates(track, im_size)
        track_times = track[:, 2] / 30
        try:
            source_points = np.stack(
                (track[0:points - 1, 0], track[0:points - 1, 1], track_times[0:points - 1]),
                axis=1
            )
            apex_index = find_apex_index(track)
            last_point = [track[-1, 0], track[-1, 1], track_times[-1]]
            if (mode == FittingMode.ApexPoint or mode == FittingMode.ApexAndLast) and apex_index >= points:
                apex_point = [track[apex_index, 0], track[apex_index, 1], track_times[apex_index]]
                source_points[-1] = apex_point
            if mode == FittingMode.LastPoint:
                source_points[-1] = last_point
            if mode == FittingMode.ApexAndLast and source_points[-1, 2] < track_times[-1]:
                source_points = np.append(source_points, [last_point], axis=0)

            if len(source_points) < points - 1:
                print("Bad source points: {}".format(source_points))
                # continue

            start = time.time()
            if method == 'quadratic':
                result = fit_quadratic_drag(source_points, track_times)
            elif method == 'linear':
                result = fit_linear_drag(source_points, track_times)
            elif method == 'magnus':
                result = minimize_magnus(source_points, track_times, algorithm)
            else:
                print('Unexpected method {}'.format(method))
                return mean(total_dists)
            print("time: {}".format(time.time() - start))

            dists = scipy.spatial.distance.cdist(result[:, [0, 1]], track[:, [0, 1]])
            dist = sum(np.diagonal(dists)) / len(dists)
            total_dists.append(dist)
            total_tasks += 1
            print(dist)
            if visualize:
                plot(result, track[:, [0, 1]], source_points)
        except RuntimeError:
            print("Could not fit the curve for {}".format(task_path))

    return mean(total_dists)


def plot(result, track, source_points):
    fig, axs = plt.subplots(1)
    fig.suptitle('3d and camera projection')

    axs.set(xlabel='ksi', ylabel='eta')
    axs.plot(
        track[:, 0], track[:, 1], '.',
        source_points[:, 0], source_points[:, 1], '*',
        result[:, 0], result[:, 1], '-'
    )
    plt.show()


def find_apex_index(track):
    return np.argmax(track[:, 1])


def get_image_size(task_root):
    video_path = find_video(os.path.join(task_root, 'data'))
    container = av.open(video_path)

    for packet in container.demux():
        if packet.stream.type == 'video':
            for image in packet.decode():
                height, width = image.width, image.height
                if packet.stream.metadata.get('rotate'):
                    angle = 360 - int(container.streams.video[0].metadata.get('rotate'))
                    if angle == 90 or angle == 270:
                        return [height, width]
                return [width, height]
    print("Cannot read video {}".format(video_path))
    return [0, 0]


def find_video(path):
    for file in os.listdir(path):
        sub_path = os.path.join(path, file)
        if any(file.endswith(ext) for ext in video_extensions):
            return sub_path
        elif os.path.isdir(sub_path):
            return find_video(sub_path)
    return None


video_extensions = ['mov', 'mp4']

if __name__ == '__main__':
    test_mode = 'points'
    if len(sys.argv) > 2:
        test_mode = sys.argv[2]

    if test_mode == 'points':
        d1 = test_dataset(sys.argv[1], FittingMode.Normal)
        d2 = test_dataset(sys.argv[1], FittingMode.LastPoint)
        d3 = test_dataset(sys.argv[1], FittingMode.ApexAndLast)
        print("total distance: {}, with last point: {} with apex and last {}".format(round(d1), round(d2), round(d3)))
    elif sys.argv[2] == 'methods':
        algorithms = ['Nelder-Mead', 'L-BFGS-B', 'Powell', 'TNC', 'SLSQP']
        # 'CG', 'BFGS', 'Newton-CG', 'COBYLA', 'trust-constr', 'dogleg'
        # 'trust-ncg', 'trust-exact', 'trust-krylov'
        results = {}
        for alg in algorithms:
            results[alg] = test_dataset(sys.argv[1], FittingMode.ApexAndLast, False, alg)
            print('{}: {}'.format(alg, results[alg]))
        print(results)

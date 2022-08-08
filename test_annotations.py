import numpy as np
import sys
import os
import scipy
import matplotlib.pyplot as plt
from curve import fit_quadratic_drag, normalize_coordinates
from cvat_annotations import load_track
import av

from linear_drag_model import fit_linear_drag

points = 7
method = 'linear'  # 'quadratic'


def test_dataset(root, add_last=False, visualize=False):
    total_dist = 0
    total_tasks = 0
    for task in sorted(os.listdir(root)):
        if visualize and total_tasks > 5:
            break
        task_path = os.path.join(root, task)
        if not os.path.isdir(task_path):
            continue
        im_size = get_image_size(task_path)
        ann_path = os.path.join(task_path, 'annotations.json')
        track = load_track(ann_path)
        track = normalize_coordinates(track, im_size)
        track_times = track[:, 2] / 30
        try:
            print("---- {} ----".format(task))
            source_points = np.stack(
                (track[0:points - 1, 0], track[0:points - 1, 1], track_times[0:points - 1]),
                axis=1
            )
            if add_last:
                source_points[-1] = [track[-1, 0], track[-1, 1], track_times[-1]]
            if method == 'quadratic':
                result = fit_quadratic_drag(source_points, track_times)
            else:
                result = fit_linear_drag(source_points, track_times)
            dists = scipy.spatial.distance.cdist(result[:, [0, 1]], track[:, [0, 1]])
            dist = sum(np.diagonal(dists)) / len(dists)
            total_dist += dist
            total_tasks += 1
            print(dist)
            if visualize:
                plot(result, track[:, [0, 1]], source_points)
        except RuntimeError:
            print("Could not fit the curve for {}".format(task_path))

    return total_dist / total_tasks


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
    d1 = test_dataset(sys.argv[1], False, False)
    d2 = test_dataset(sys.argv[1], True, False)
    print("total distance: {}, with last point: {}".format(d1, d2))

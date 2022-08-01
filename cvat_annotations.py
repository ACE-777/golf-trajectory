import json
from functools import reduce
import pandas as pd
import numpy as np
from pandas import concat
import sys


def load_track(annotation_path):
    shapes = flatten_shapes(annotation_path)

    df = pd.DataFrame(columns=["frame", "x1", "y1", "x2", "y2", "occluded"], dtype=object)
    last_frame = int(shapes[0]['frame']) - 1
    total_skipped = 0
    seen = set()
    for shape in shapes:
        frame = int(shape['frame'])

        frames_skipped = frame - last_frame - 1
        if frames_skipped > 0:
            total_skipped = total_skipped + frames_skipped
            # Add empty rows for skipped frames
            for i in range(0, frames_skipped):
                f = i + last_frame + 1
                if f in seen:
                    print('skipped frame was already handled! {}'.format(f))
                    continue
                seen.add(f)
                df = concat([df, pd.DataFrame(
                    [[f, np.nan, np.nan, np.nan, np.nan, np.nan]],
                    columns=["frame", "x1", "y1", "x2", "y2", "occluded"]
                )], ignore_index=True)

        if frame in seen:
            print('frame was already handled! {}'.format(frame))
            continue
        seen.add(frame)

        occluded = is_occluded(shape)

        df = concat([df, pd.DataFrame([[
            frame,
            float(shape['points'][0]),
            float(shape['points'][1]),
            float(shape['points'][2]),
            float(shape['points'][3]),
            occluded
        ]], columns=["frame", "x1", "y1", "x2", "y2", "occluded"]
        )], ignore_index=True)
        last_frame = frame

    df['occluded'] = df['occluded'].fillna(True)
    df["frame"] = pd.to_numeric(df["frame"])
    df["x1"] = pd.to_numeric(df["x1"])
    df["y1"] = pd.to_numeric(df["y1"])
    df["x2"] = pd.to_numeric(df["x2"])
    df["y2"] = pd.to_numeric(df["y2"])
    df.interpolate(inplace=True)
    df.fillna(method='bfill', inplace=True)

    track = []
    last_occluded = 0
    for r in df.iloc[:len(df['occluded']), :].values:
        occluded = int(r[5])
        if occluded == 1:
            last_occluded += 1
        else:
            last_occluded = 0
        x = r[1]
        y = r[2]
        w = r[3] - x
        h = r[4] - y
        track.append([x + w / 2, y + h / 2])
    print("removing occluded from the end: {}".format(last_occluded))
    del track[-last_occluded]

    print('Added empty rows for {} frames'.format(total_skipped))
    return track


def is_occluded(shape):
    return bool(shape['occluded']) or bool(shape.get('outside', False))


def merge_shapes(list1, list2):
    not_occluded_seen = set(map(lambda s: s['frame'], filter(lambda s: not is_occluded(s), list1)))
    seen = set(map(lambda s: s['frame'], list1))
    for shape in list2:
        frame_num = shape['frame']
        if frame_num in not_occluded_seen and is_occluded(shape):
            print('Skipping {} duplicated occluded annotation'.format(frame_num))
        elif frame_num not in seen:
            list1.append(shape)
            seen.add(frame_num)
            if not is_occluded(shape):
                not_occluded_seen.add(frame_num)
    return list1


def flatten_shapes(annotation_path):
    with open(annotation_path) as f:
        data = json.load(f)[0]
        tracks = data['tracks']
        tracks.sort(key=lambda t: t['frame'])
        if len(tracks) == 0:
            return data['shapes']
        else:
            shapes = map(lambda t: t['shapes'], tracks)
        return reduce(merge_shapes, shapes, [])


if __name__ == '__main__':
    print(load_track(sys.argv[1]))

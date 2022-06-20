import cv2
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import csv
import pandas as pd
import argparse
import colorsys


def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Reproject')
  
  parser.add_argument('--video_path', dest='video_path',
                      help='video file path',
                      default="", type=str)
  parser.add_argument('--out_path', dest='out_path',
                      help='output video file path',
                      default="", type=str)
  
  args = parser.parse_args()
  return args


def hsv2rgb(h,s,v):
    return tuple(round(i * 255) for i in colorsys.hsv_to_rgb(h,s,v))


class Reprojector:
    
    def __init__(self, path, video_path, out_path):
        """
        import tracks and bbox
        """
        self.tracks = []
        self.diameter = 11 #cm
        self.focal = 28 #mm
        self.imgs = []
        self.video_path = video_path
        self.out_path = out_path

        
        self.df = pd.read_csv('trajectory.csv', index_col='frame')

        with open('output.txt') as f:
            f.readline()
            lines = f.readlines()
            for line in lines:
                vals = line.split(',')
                bbox = []
                for v in vals[1:5]:
                    bbox.append(int(float(v)))
                track = vals[5] == 'True'
                bbox[0] = bbox[0] + bbox[2] / 2
                bbox[1] = bbox[1] + bbox[3] / 2
                bbox.append(track)
                bbox.append(int(vals[0]))
                bbox.append(int(vals[6]))
                self.tracks.append(bbox)

            count = len(self.tracks)
            for i in reversed(range(count)):
                bbox = self.tracks[i]
                if bbox[4]:
                    break
                del self.tracks[i]
        
        with open(path + 'cam.txt') as f:
            lines = f.readlines()
            vals = []
            for line in lines:
                vals.append(float(line.split()[0]))
            self.cx = vals[0]
            self.cy = vals[1]
            self.diameter = vals[2]
            self.focal = vals[3]
    
    def project(self, point):
        # focal = 28mm
        #fovy = 65.5
        #fovx = 46.4
        fovd = 75.4

        width = self.cx * 2
        height = self.cy * 2

        diag_ratio = math.tan(math.radians(fovd) / 2) * 2
        width_ratio = width / math.sqrt(width*width + height*height) * diag_ratio
        fovx = math.degrees(math.atan(width_ratio / 2) * 2)
        
        # [x]   [fx 0 cx][X]
        # [y] = [0 fy cy][Y]
        # [z]   [0 0  1 ][Z]
        fx = fy = width / (math.tan(math.radians(fovx) / 2) * 2)

        x = fx*point[0] + self.cx*point[2]
        y = -fy*point[1] + self.cy*point[2]
        z = point[2]

        x = x / z
        y = y / z

        return (x,y,z)

    def draw(self, img, row):
        cam_pt = self.project((row['x'], row['y'], row['z']))
        x = cam_pt[0]
        y = cam_pt[1]

        # Blue color in BGR
        color = (255, 0, 0)
        
        # Line thickness of 2 px
        thickness = 2

        w = 10
        h = 10
        
        img = cv2.rectangle(img,
            (int(x-w/2),int(y-h/2)),
            (int(x+w/2), int(y+h/2)),
            color, thickness)

        return img

    def draw_curve(self, img):
        last_pt = None
        thickness = 2

        start = 0.1
        end = 0.9

        samples = 50
        count = self.df.shape[0]
        skip = count / samples

        index = 0
        for _, row in self.df.iterrows():
            h = start + index / count *(end-start)
            index = index + 1

            if (index-1) % skip != 0:
                continue

            color = hsv2rgb(h,1,1)
            cur_pt = self.project((row['x'], row['y'], row['z']))
            
            if last_pt is not None:
                img = cv2.line(img,
                    (int(last_pt[0]),int(last_pt[1])),
                    (int(cur_pt[0]), int(cur_pt[1])),
                    color, thickness)
            
            last_pt = cur_pt

        return img

    def play(self):
        frame_id = 0

        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        out = cv2.VideoWriter(self.out_path, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (frame_width,frame_height))

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            last_frame = frame.copy()
            last_frame = cv2.flip(last_frame, 0)
            last_frame = cv2.flip(last_frame, 1)

            frame_id = frame_id + 1
            draw_img = last_frame.copy()

            draw_img = self.draw_curve(draw_img)

            if frame_id in self.df.index:
                row = self.df.loc[frame_id]
                draw_img = self.draw(draw_img, row)
            
            cv2.imshow("frame", draw_img)
            cv2.waitKey(33)

            out.write(draw_img)

        while frame_id in self.df.index:
            frame_id = frame_id + 1
            draw_img = last_frame.copy()

            draw_img = self.draw_curve(draw_img)

            if frame_id in self.df.index:
                row = self.df.loc[frame_id]
                self.draw(draw_img, row)
            
            cv2.imshow("frame", draw_img)
            cv2.waitKey(33)

            out.write(draw_img)

        cap.release()
        out.release()

    def export(self):
        f = open("trajectory.csv", "w")
        f.write('frame,x,y,z\n')
        for pt in self.trajectory:
            t = int(pt[0]+self.first_frame)
            world_pt = (pt[1]+self.first_pt[0], pt[2]+self.first_pt[1], pt[3]+self.first_pt[2])
            f.write(f'{t},{world_pt[0]:.6f},{world_pt[1]:.6f},{world_pt[2]:.6f}\n')
        f.close()

    def eval(self):
        count = 0
        sum = 0

        for pt in self.tracks:
            frame_id = pt[5]
            if not frame_id in self.df.index:
                continue

            row = self.df.loc[frame_id]
            cam_pt = self.project((row['x'], row['y'], row['z']))
            x = cam_pt[0]
            y = cam_pt[1]

            x_ = pt[0]
            y_ = pt[1]

            sum = sum + math.sqrt((x-x_)*(x-x_) + (y-y_)*(y-y_))
            
            count = count + 1

        if count == 0:
            residual = 0
        else:
            residual = sum / count
        print('residual', residual)
        return residual


if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    reproj = Reprojector('', args.video_path, args.out_path)
    #reconstructor.fit()
    reproj.play()
    reproj.eval()
    #reconstructor.dump()
    #reconstructor.plot()
    #reproj.export()

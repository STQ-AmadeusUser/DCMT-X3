import os
import cv2
from glob import glob


def get_frames(video_name):
    if video_name == '':
        cap = cv2.VideoCapture(0)
        # warmup
        for i in range(5):
            cap.read()
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break
    elif video_name.endswith('avi') or video_name.endswith('mp4') or video_name.endswith('flv'):
        cap = cv2.VideoCapture(video_name)
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break
    else:
        images = glob(os.path.join(video_name, '*.jp*'))
        images = sorted(images, key=lambda x: int(x.split('/')[-1].split('.')[0]))
        for img in images:
            frame = cv2.imread(img)
            yield frame
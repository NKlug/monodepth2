from kitti_utils import load_oxts, get_absolute_camera_orientation, generate_depth_map
import os
import numpy as np
import cv2

if __name__ == '__main__':
    data_dir = '/home/nikolas/Datasets/MannequinChallenge/MannequinChallenge/'

    mp4_path = os.path.join(data_dir, 'validation', '00c9878266685887.mp4')
    out_folder = os.path.join(data_dir, 'tmp')

    vidcap = cv2.VideoCapture(mp4_path)
    success, image = vidcap.read()
    count = 0
    print(vidcap.get(cv2.CAP_PROP_FPS))
    while success:
        # image_file_path = os.path.join(out_folder, 'img_{:0>6d}.jpg'.format(count))
        count += 1
        # cv2.imwrite(image_file_path, image)
        success, image = vidcap.read()
    print(f"total number of frames: {count}")

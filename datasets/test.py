from kitti_utils import load_oxts, get_absolute_camera_orientation, generate_depth_map
import os
import numpy as np
import cv2

if __name__ == '__main__':
    calib_dir = '/home/nikolas/Datasets/kitti_data/'
    oxts_filename = '0000000000.txt'
    velo_filename = '0000000000.bin'

    velo_filename = os.path.join(calib_dir, '2011_09_26_drive_0001_sync/velodyne_points/data', velo_filename)
    depth_map = generate_depth_map(calib_dir, velo_filename)
    # depth_map = depth_map / np.max(depth_map) * 255
    #
    # cv2.imshow('asdf', depth_map.astype(np.uint8))
    # cv2.waitKey(0)

    get_absolute_camera_orientation(calib_dir, None)
    pass

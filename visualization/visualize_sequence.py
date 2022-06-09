import os

from torch.utils.data import DataLoader

import datasets
import networks
from kitti_utils import read_calib_file
from trainer import Trainer
from utils import readlines
import torch
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import matplotlib.cm as cm
import cv2
import pickle

from layers import BackprojectDepth
from options import MonodepthOptions

splits_dir = os.path.join(os.path.dirname(__file__), "splits")


def back_project_depths(data, opt):
    """
    Compute 3d coordinates by perspective re-projection using the 2d coordinates and estimated depths.
    """
    # back-project as during training
    depths = data["depth"]
    h, w = depths.shape[1:]
    back_project = BackprojectDepth(opt.batch_size, h, w).cuda()

    cam_points = []
    with torch.no_grad():
        batch_size = opt.batch_size
        for i in range(np.ceil(len(depths) / batch_size).astype(np.int)):
            depths_batch = torch.from_numpy(depths[i * batch_size:(i + 1) * batch_size]).cuda()
            inv_K_batch = torch.from_numpy(data["inv_K"][i * batch_size:(i + 1) * batch_size]).cuda()
            cam_points_batch = back_project(depths_batch, inv_K_batch).cpu().numpy()
            cam_points_batch = cam_points_batch.reshape((-1, 4, 192, 640))
            cam_points_batch = cam_points_batch[:, :3, ...]

            cam_points.append(cam_points_batch)

    cam_points = np.concatenate(cam_points)

    data["cam_points"] = cam_points
    return data


def back_project_depth(predicted_depth, inv_K):
    """
    Compute 3d coordinates by perspective re-projection using the 2d coordinates and estimated depths.
    """
    # back-project as during training
    h, w = predicted_depth.shape
    back_project = BackprojectDepth(1, h, w).cpu()

    predicted_depth = predicted_depth[np.newaxis, ...]
    inv_K = inv_K[np.newaxis, ...]

    with torch.no_grad():
        depths_batch = torch.from_numpy(predicted_depth).cpu()
        inv_K_batch = torch.from_numpy(inv_K).cpu()
        cam_points = back_project(depths_batch, inv_K_batch).cpu().numpy()
        cam_points = cam_points.reshape((-1, 4, h, w))
        cam_points = cam_points[:, :3, ...]

    return cam_points[0].T


def get_image_to_imu_matrix(calib_dir, cam=2):
    """Compute image to imu homogenous transformation matrix."""
    # load calibration files
    imu2velo = read_calib_file(os.path.join(calib_dir, 'calib_imu_to_velo.txt'))
    cam2cam = read_calib_file(os.path.join(calib_dir, 'calib_cam_to_cam.txt'))
    velo2cam = read_calib_file(os.path.join(calib_dir, 'calib_velo_to_cam.txt'))

    # velo2cam (velo2rect) matrix
    velo2cam = np.hstack((velo2cam['R'].reshape(3, 3), velo2cam['T'][..., np.newaxis]))
    velo2cam = np.vstack((velo2cam, np.array([0, 0, 0, 1.0])))

    # imu2velo matrix
    imu2velo = np.hstack((imu2velo['R'].reshape(3, 3), imu2velo['T'][..., np.newaxis]))
    imu2velo = np.vstack((imu2velo, np.array([0, 0, 0, 1.0])))

    # cam2rect matrix
    R_cam2rect = np.eye(4)
    R_cam2rect[:3, :3] = cam2cam['R_rect_00'].reshape(3, 3)

    # rect2camX matrix
    P_rect = np.eye(4)
    P_rect[:3, :4] = cam2cam['P_rect_0' + str(cam)].reshape(3, 4)

    P_velo2im = np.dot(np.dot(P_rect, R_cam2rect), velo2cam)
    P_imu2im = np.dot(P_velo2im, imu2velo)

    # invert to get im2imu
    return np.linalg.inv(P_imu2im)


def experimental_compute_3d_coordinates(predicted_depth, inv_K, in_imu_coordinates=False, f=5, interim_downscale=4):
    orig_w, orig_h = 1242, 375

    h, w = predicted_depth.shape[:2]
    predicted_depth = cv2.resize(predicted_depth, (w // interim_downscale, h // interim_downscale))
    h, w = predicted_depth.shape[:2]
    #
    x = np.linspace(-orig_w, orig_w, w)
    y = np.linspace(-orig_h, orig_h, h)
    xv, yv = np.meshgrid(x, y)

    coords_2d = np.stack([xv, yv], axis=-1)
    coords_3d = np.concatenate([coords_2d, predicted_depth[..., np.newaxis]], axis=-1)


    # re-project
    # xv = f * xv
    # yv = f * yv

    # re-project as during training
    # coords_3d = back_project_depth(predicted_depth, inv_K)

    # upscale to original image size
    #
    # coords_3d[:, :, 0] *= orig_w
    # coords_3d[:, :, 1] *= orig_h

    if in_imu_coordinates:
        h, w = coords_3d.shape[:2]
        coords_3d = np.concatenate([coords_3d, np.ones((h, w, 1))], axis=-1)
        coords_3d = coords_3d.reshape((-1, 4))

        calib_dir = '/home/nikolas/Datasets/kitti_data/'
        M = get_image_to_imu_matrix(calib_dir)

        # transform to imu coords
        coords_3d = np.dot(M, coords_3d.T).T

        coords_3d = coords_3d[:, :3].reshape((h, w, 3))

    return coords_3d
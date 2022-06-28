import os

import cv2
import numpy as np
import torch

from layers import BackprojectDepth

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


def experimental_compute_3d_coordinates(predicted_depth, inv_K, f=5, downscale=4, *args, **kwargs):
    orig_w, orig_h = 1242, 375

    h, w = predicted_depth.shape[:2]
    predicted_depth = cv2.resize(predicted_depth, (w // downscale, h // downscale))
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

    return coords_3d

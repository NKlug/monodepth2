import os

from torch.utils.data import DataLoader

import datasets
from utils import readlines
import torch
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import cv2
import pickle

from layers import BackprojectDepth
from options import MonodepthOptions

splits_dir = os.path.join(os.path.dirname(__file__), "splits")


def back_project_depths(data, opt):
    # back-project as during training
    depths = data["depth"]
    h, w = depths.shape[1:]
    back_project = BackprojectDepth(opt.batch_size, h, w).cuda()

    cam_points = []
    with torch.no_grad():
        batch_size = opt.batch_size
        for i in range(np.ceil(len(depths)/batch_size).astype(np.int)):
            depths_batch = torch.from_numpy(depths[i * batch_size:(i+1) * batch_size]).cuda()
            inv_K_batch = torch.from_numpy(data["inv_K"][i * batch_size:(i + 1) * batch_size]).cuda()
            cam_points_batch = back_project(depths_batch, inv_K_batch).cpu().numpy()
            cam_points_batch = cam_points_batch.reshape((-1, 4, 192, 640))
            cam_points_batch = cam_points_batch[:, :3, ...]

            cam_points.append(cam_points_batch)

    cam_points = np.concatenate(cam_points)

    data["cam_points"] = cam_points
    return data


def visualize_sequence(data, absolute_coordinates):

    # TODO: Maybe use true camera intrinsics
    f = 5
    interim_downscale = 2
    predicted_depths = data["cam_points"][:1]

    for predicted_depth, absolute_coordinate in zip(predicted_depths, absolute_coordinates):
        # h, w = predicted_depth.shape[:2]
        # predicted_depth = cv2.resize(predicted_depth, (w//interim_downscale, h//interim_downscale))
        h, w = predicted_depth.shape[:2]

        # aspect_ratio = h/w
        # x = np.linspace(-1, 1, w)
        # y = np.linspace(-1*aspect_ratio, 1*aspect_ratio, h)
        # xv, yv = np.meshgrid(x, y)
        #
        # # flip y coordinates
        # yv = yv[::-1]
        #
        # # re-project
        # # xv = f * xv * predicted_depth
        # # yv = f * yv * predicted_depth
        # xv = f * xv
        # yv = f * yv

        xv = predicted_depth[0, ...]
        yv = predicted_depth[1, ...]
        zv = predicted_depth[2, ...]

        image_coords = predicted_depth[:2, ...]

        # predicted_depth = 30 * predicted_depth

        vmax = np.percentile(zv, 95)
        normalizer = mpl.colors.Normalize(vmin=yv.min(), vmax=vmax)
        mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
        colormapped_im = mapper.to_rgba(zv)[:, :, :3]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        ax.scatter(xv.flatten(), yv.flatten(), zv.flatten(), s=1, c=colormapped_im.reshape((-1, 3)))
        plt.show()

    pass


if __name__ == '__main__':
    options = MonodepthOptions()
    opt = options.parse()

    output_path = os.path.join(
        opt.load_weights_folder, "predicted_depths_{}_split.pkl".format(opt.eval_split))
    print("-> Loading predicted depths from ", output_path)
    with open(output_path, 'rb') as f:
        data = pickle.load(f)

    camera_coordinates = [(0, 0, 1, 0, 0, 0)]

    data = back_project_depths(data, opt)

    visualize_sequence(data, camera_coordinates)

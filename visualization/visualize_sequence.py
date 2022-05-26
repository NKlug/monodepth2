import os
import torch
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import cv2

from layers import BackprojectDepth
from options import MonodepthOptions


def back_project_depths(predicted_depths, opt):
    # back-project as during training
    h, w = predicted_depths.shape[1:]
    back_project = BackprojectDepth(opt.batch_size, h, w).cuda()
    with torch.no_grad():
        for i in range(opt.batch_size):
            pred_depths = torch.
            cam_points = back_project(predicted_depths[i * batchsize:(i+1)*batch_size], inputs[("inv_K", source_scale)])


def visualize_sequence(predicted_depths, absolute_coordinates):

    # TODO: Maybe use true camera intrinsics
    f = 5
    interim_downscale = 2

    for predicted_depth, absolute_coordinate in zip(predicted_depths, absolute_coordinates):
        h, w = predicted_depth.shape[:2]
        predicted_depth = cv2.resize(predicted_depth, (w//interim_downscale, h//interim_downscale))
        h, w = predicted_depth.shape[:2]

        aspect_ratio = h/w
        x = np.linspace(-1, 1, w)
        y = np.linspace(-1*aspect_ratio, 1*aspect_ratio, h)
        xv, yv = np.meshgrid(x, y)

        # flip y coordinates
        yv = yv[::-1]

        # re-project
        # xv = f * xv * predicted_depth
        # yv = f * yv * predicted_depth
        xv = f * xv
        yv = f * yv

        predicted_depth = 30 * predicted_depth

        vmax = np.percentile(predicted_depth, 95)
        normalizer = mpl.colors.Normalize(vmin=predicted_depth.min(), vmax=vmax)
        mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
        colormapped_im = mapper.to_rgba(predicted_depth)[:, :, :3]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        ax.scatter(xv.flatten(), yv.flatten(), predicted_depth.flatten(), s=1, c=colormapped_im.reshape((-1, 3)))
        plt.show()

    pass


if __name__ == '__main__':
    options = MonodepthOptions()
    opt = options.parse()

    output_path = os.path.join(
        opt.load_weights_folder, "disps_{}_split.npy".format(opt.eval_split))
    print("-> Loading predicted depths from ", output_path)
    predicted_depths = np.load(output_path)

    predicted_depths = predicted_depths[:1]
    camera_coordinates = [(0, 0, 1, 0, 0, 0)]

    visualize_sequence(predicted_depths, camera_coordinates)

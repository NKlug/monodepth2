import os

from torch.utils.data import DataLoader

import datasets
import networks
from kitti_utils import get_absolute_camera_orientation
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


def estimate_relative_pose(image1, image2, opt):
    """
    Estimate relative camera pose between images using the pose network with learned weights.
    """
    opt.load_weights_folder = os.path.expanduser(opt.load_weights_folder)

    assert os.path.isdir(opt.load_weights_folder), \
        "Cannot find a folder at {}".format(opt.load_weights_folder)

    print("-> Loading weights from {}".format(opt.load_weights_folder))

    pose_encoder_path = os.path.join(opt.load_weights_folder, "pose_encoder.pth")
    pose_decoder_path = os.path.join(opt.load_weights_folder, "pose.pth")

    pose_encoder_dict = torch.load(pose_encoder_path)

    filenames = readlines(os.path.join(splits_dir, opt.eval_split, "test_files.txt"))
    # noinspection DuplicatedCode
    data = datasets.KITTIRAWDataset(options.data_path, filenames,
                                    pose_encoder_dict['height'], pose_encoder_dict['width'],
                                    [0], 4, is_train=False)
    dataloader = DataLoader(data, 16, shuffle=False, num_workers=opt.num_workers,
                            pin_memory=True, drop_last=False)

    pose_encoder = networks.ResnetEncoder(opt.num_layers, False)
    pose_decoder = networks.PoseDecoder(pose_encoder.num_ch_enc, num_input_features=1, num_frames_to_predict_for=2)

    model_dict = pose_encoder.state_dict()
    pose_encoder.load_state_dict({k: v for k, v in pose_encoder_dict.items() if k in model_dict})
    pose_decoder.load_state_dict(torch.load(pose_decoder_path))

    pose_encoder.cuda()
    pose_encoder.eval()
    pose_decoder.cuda()
    pose_decoder.eval()



    pose_inputs = [pose_encoder(torch.cat(pose_inputs, 1))]

    axisangle, translation = self.models["pose"](pose_inputs)
    outputs[("axisangle", 0, f_i)] = axisangle
    outputs[("translation", 0, f_i)] = translation

    # Invert the matrix if the frame id is negative
    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
        axisangle[:, 0], translation[:, 0], invert=(f_i < 0))


def get_camera_rotation_matrix(oxts):
    """
    Computes camera rotation matrices for leveling predicted 3d coordinates.
    """
    for i in range(len(oxts['lat'])):
        pass


def compute_3d_coordinates(predicted_depth, f=5, interim_downscale=4):
    h, w = predicted_depth.shape[:2]
    predicted_depth = cv2.resize(predicted_depth, (w // interim_downscale, h // interim_downscale))
    h, w = predicted_depth.shape[:2]

    aspect_ratio = h / w
    x = np.linspace(-1, 1, w)
    y = np.linspace(-1 * aspect_ratio, 1 * aspect_ratio, h)
    xv, yv = np.meshgrid(x, y)

    # flip y coordinates
    yv = yv[::-1]
    #
    # # re-project
    # # xv = f * xv * predicted_depth
    # # yv = f * yv * predicted_depth
    xv = f * xv
    yv = f * yv

    # xv = predicted_depth[0, ...]
    # yv = predicted_depth[1, ...]
    # zv = predicted_depth[2, ...]

    zv = predicted_depth
    coordinates_3d = np.stack([xv.flatten(), yv.flatten(), zv.flatten()], axis=-1)

    # calib_dir = '/home/nikolas/Datasets/kitti_data/'
    # P = get_absolute_camera_orientation(calib_dir, None)
    # for i in range(len(coordinates_3d)):
    #     coordinates_3d[i] = np.dot(P, np.concatenate([coordinates_3d[i], np.ones((1,))]).T).T

    coordinates_3d = coordinates_3d.reshape((h, w, 3))
    return coordinates_3d


def visualize_single_step(data):
    predicted_depths = data["disp"][:1]

    coords = []
    for predicted_depth in predicted_depths:
        coords_3d = compute_3d_coordinates(predicted_depth, interim_downscale=8)
        coords.append(coords_3d)

    coords = np.asarray(coords)

    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(111, projection='3d', proj_type='persp')
    # ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    for i in range(len(coords)):
        xv = coords[i, :, :, 0]
        yv = coords[i, :, :, 1]
        zv = coords[i, :, :, 2]
        pred_depth = predicted_depths[i]
        vmax = np.percentile(pred_depth, 95)
        normalizer = mpl.colors.Normalize(vmin=pred_depth.min(), vmax=vmax)
        mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
        h, w = pred_depth.shape[:2]
        colormapped_im = mapper.to_rgba(cv2.resize(pred_depth, (w // 8, h // 8)))[:, :, :3]
        scattered = ax.scatter(xv.flatten(), yv.flatten(), zv.flatten(), s=200, c=colormapped_im.reshape((-1, 3)))
        MAX = 3
        for direction in (-1, 1):
            for point in np.diag(direction * MAX * np.array([1, 1, 1])):
                ax.plot([point[0]], [point[1]], [point[2]], 'w')
        #
        # import plotly
        # import plotly.graph_objs as go
        #
        # # Configure the trace.
        # trace = go.Scatter3d(
        #     x=xv.flatten(),  # <-- Put your data instead
        #     y=yv.flatten(),  # <-- Put your data instead
        #     z=zv.flatten(),  # <-- Put your data instead
        #     # color_discrete_sequence=colormapped_im.reshape((-1, 3)),
        #     mode='markers',
        #     marker={
        #         'size': 10,
        #         'opacity': 0.8,
        #     }
        # )
        #
        # # Configure the layout.
        # layout = go.Layout(
        #     margin={'l': 0, 'r': 0, 'b': 0, 't': 0}
        # )
        #
        # data = [trace]
        #
        # fig = go.Figure(data=data, layout=layout)

    # def next_scatter(num, coords):
    #     xv = coords[num, :, :, 0]
    #     yv = coords[num, :, :, 1]
    #     zv = coords[num, :, :, 2]
    #     vmax = np.percentile(zv, 95)
    #     normalizer = mpl.colors.Normalize(vmin=zv.min(), vmax=vmax)
    #     mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
    #     colormapped_im = mapper.to_rgba(zv)[:, :, :3]
    #     ax.clear()
    #     scattered = ax.scatter(xv.flatten(), yv.flatten(), zv.flatten(), s=200, c=colormapped_im.reshape((-1, 3)))
    #     # scattered = ax.plot_surface(xv, yv, zv, cmap='coolwarm', linewidth=1, antialiased=True)
    #     MAX = 5
    #     # for direction in (-1, 1):
    #     #     for point in np.diag(direction * MAX * np.array([1, 1, 1])):
    #     #         ax.plot([point[0]], [point[1]], [point[2]], 'w')
    #
    #     # scattered = ax.scatter(xv.flatten(), yv.flatten(), zv.flatten(), s=2)
    #     # for line, data in zip(lines, dataLines):
    #     #     # NOTE: there is no .set_data() for 3 dim data...
    #     #     line.set_data(data[0:2, :num])
    #     #     line.set_3d_properties(data[2, :num])
    #     # scatter_plots[num-1].set_visible(False)
    #     # scatter_plots[num].set_visible(True)
    #
    # line_ani = animation.FuncAnimation(fig, next_scatter, len(coords), fargs=(coords, ), interval=100, blit=False)

    plt.show()

    # fig.show()

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

    # data = {"disp": np.load('/home/nikolas/Projects/monodepth2/assets/test_image_disp.npy')[:, 0]}

    # data = back_project_depths(data, opt)

    simple_visualize_sequence(data)

import cv2
import numpy as np
import scipy.spatial as spat
import torch

from kitti_utils import get_image_to_imu_matrix
from layers import BackprojectDepth, disp_to_depth
from utils import lat_lon_to_meters


def get_global_coords(data):
    lat = data['oxts']['lat']
    lon = data['oxts']['lon']
    alt = data['oxts']['alt']

    roll = data['oxts']['roll']
    pitch = data['oxts']['pitch']
    yaw = data['oxts']['yaw']

    lat, lon = lat_lon_to_meters(lat, lon)
    lat = lat - lat[0]
    lon = lon - lon[0]
    alt = alt - alt[0] + 1

    return lat, lon, alt, roll, pitch, yaw


def compute_3d_coordinates(data, downscale=1, global_coordinates=False, max_depth=None):
    """
    Compute 3d coordinates from predicted depth and camera intrinsics in meters.
    """
    back_project_fn = back_project_perspective

    predicted_depths = data["depth"]
    lat, lon, alt, roll, pitch, yaw = get_global_coords(data)

    coords = []
    for i, predicted_depth in enumerate(predicted_depths):

        if max_depth is not None:
            predicted_depth = np.minimum(predicted_depth, max_depth)

        inv_K = data['inv_K'][i]
        coords_3d = back_project_fn(predicted_depth, downscale=downscale, inv_K=inv_K)

        if global_coordinates:
            coords_3d = image_to_imu_coordinates(coords_3d)

            # compute coordinates in global coordinate system
            position = np.asarray([lat[i], lon[i], alt[i]])
            orientation = np.asarray([roll[i], pitch[i], yaw[i]])
            rot = spat.transform.Rotation.from_euler('xyz', orientation).as_matrix()

            global2imu = np.eye(4)
            global2imu[:3, :3] = rot
            global2imu[:3, 3] = position
            imu2global = np.linalg.inv(global2imu)
            imu2global = global2imu

            h, w = coords_3d.shape[:2]
            coords_3d = np.concatenate([coords_3d, np.ones((h, w, 1))], axis=-1)
            coords_3d = coords_3d.reshape((-1, 4))

            coords_3d = np.dot(imu2global, coords_3d.T).T

            coords_3d = coords_3d[:, :3]
            coords_3d = coords_3d.reshape((h, w, 3))

        coords.append(coords_3d)

    return np.asarray(coords)


def back_project_orthographic(predicted_depth, f=5, downscale=4, *args, **kwargs):
    """
    Back project predicted depths to 3D space.
    """
    h, w = predicted_depth.shape[:2]
    predicted_depth = cv2.resize(predicted_depth, (w // downscale, h // downscale), interpolation=cv2.INTER_LINEAR)
    h, w = predicted_depth.shape[:2]

    aspect_ratio = h / w
    x = np.linspace(-1, 1, w)
    y = np.linspace(-1 * aspect_ratio, 1 * aspect_ratio, h)
    xv, yv = np.meshgrid(x, y)

    # flip y coordinates
    # yv = yv[::-1]

    # # re-project
    # # xv = f * xv * predicted_depth
    # # yv = f * yv * predicted_depth
    xv = f * xv
    yv = f * yv

    zv = predicted_depth
    coordinates_3d = np.stack([xv.flatten(), yv.flatten(), zv.flatten()], axis=-1)
    coordinates_3d = coordinates_3d.reshape((h, w, 3))

    return coordinates_3d


def back_project_perspective(predicted_depth, inv_K, downscale=4, *args, **kwargs):
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

    cam_points = cam_points[0].T
    cam_points = cv2.resize(cam_points, (h // downscale, w // downscale), interpolation=cv2.INTER_LINEAR)

    return cam_points


def image_to_imu_coordinates(coords_3d, calib_dir='/home/nikolas/Datasets/kitti_data/'):
    """Transform given 3d coordinates in image coordinate system to coordinates in imu coordinate system"""
    h, w = coords_3d.shape[:2]
    coords_3d = np.concatenate([coords_3d, np.ones((h, w, 1))], axis=-1)
    coords_3d = coords_3d.reshape((-1, 4))

    M = get_image_to_imu_matrix(calib_dir)

    # transform to imu coords
    coordinates_3d = np.dot(M, coords_3d.T).T

    return coordinates_3d[:, :3].reshape((h, w, 3))

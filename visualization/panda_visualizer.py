from direct.showbase.ShowBase import ShowBase
from direct.showbase.ShowBaseGlobal import globalClock
from direct.task import Task
from direct.showbase.Loader import Loader
from panda3d.core import *
import numpy as np
import cv2
import matplotlib as mpl
import matplotlib.cm as cm
import scipy.spatial as spat

from visualization.compute_3d_coordinates import compute_3d_coordinates
from visualization.controllable_show_base import ControllableShowBase

SIZE = 800
FORWARD = 'forward'
BACKWARD = 'backward'
LEFT = 'left'
RIGHT = 'right'
UP = 'up'
DOWN = 'down'

ON = 1
OFF = 0


def turn_pose(pose):
    """
    First switches y and z components and the invert the z component to make the pose stand upright,
    where the head is in positive z direction.
    :param pose:
    :return:
    """
    if len(pose.shape) == 2:
        pose = pose[:, [0, 2, 1]] * np.array([1, 1, -1])
    elif len(pose.shape) == 3:
        pose = pose[:, :, [0, 2, 1]] * np.array([1, 1, -1])
    elif len(pose.shape) == 4:
        pose = pose[:, :, :, [0, 2, 1]] * np.array([1, 1, -1])
    else:
        raise Exception("Not implemented {}".format(len(pose.shape)))
    return pose


class Visualizer(ControllableShowBase):

    def __init__(self, data):
        ControllableShowBase.__init__(self)

        self.data = data
        self.base_sphere_scale = 0.01

        self.root = NodePath('Depth 0')

        axes_node = self.create_axes_and_grid(length=20)
        axes_node.reparentTo(self.root)

        self.depth_node = None

        self.step_num = 0
        self.downscale = 6
        self.predicted_depths = self.data['depth']
        self.coords_3d = compute_3d_coordinates(self.data, downscale=self.downscale, global_coordinates=True)
        self.colors = np.asarray([self.compute_coloring(d, self.downscale) for d in self.predicted_depths])

    def compute_coloring(self, depths, downscale=1):
        h, w = depths.shape[:2]
        depths = cv2.resize(depths, (w // downscale, h // downscale))

        vmax = np.percentile(depths, 95)
        normalizer = mpl.colors.Normalize(vmin=depths.min(), vmax=vmax)
        mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
        colors = mapper.to_rgba(depths)[:, :, :3]
        return np.swapaxes(colors, 0, 1)

    def next_step(self):
        self.step_num = (self.step_num + 1) % len(self.coords_3d)

        self.depth_node.removeNode()
        self.depth_node = self._render_single_depth_map()
        self.depth_node.reparentTo(self.root)

    def previous_step(self):
        self.step_num = (self.step_num - 1) % len(self.coords_3d)

        self.depth_node.removeNode()
        self.depth_node = self._render_single_depth_map()
        self.depth_node.reparentTo(self.root)

    def visualize_multi_step(self, step_num=0):
        self.step_num = step_num

        self.accept('n', self.next_step)
        self.accept('n-repeat', self.next_step)
        self.accept('b', self.previous_step)
        self.accept('b-repeat', self.previous_step)

        pose_node = self._render_three_depth_maps()


        pose_node.reparentTo(self.root)
        self.camera.reparentTo(self.root)
        self.setBackgroundColor(200, 200, 200)        # swap z and y axis

    def visualize_single_step(self, step_num=0):
        """
        Create simple animation of back projected 3D points without accounting for relative position change.
        """
        self.step_num = step_num

        self.accept('n', self.next_step)
        self.accept('n-repeat', self.next_step)
        self.accept('b', self.previous_step)
        self.accept('b-repeat', self.previous_step)

        pose_node = self._render_single_depth_map()

        pose_node.reparentTo(self.root)
        self.camera.reparentTo(self.root)
        self.setBackgroundColor(200, 200, 200)        # swap z and y axis
        # coords = coords[..., [0, 2, 1]]
        # coords[..., 2] *= -1

    def _render_single_depth_map(self, use_relative_depths=False):
        """
        Renders the given 3d coordinates as a point cloud.
        """
        coords_3d = self.coords_3d[self.step_num]
        # compute colors
        relative_depths = self.predicted_depths[self.step_num]
        colors = self.colors[self.step_num]

        self.depth_node = NodePath('depth base node')

        h, w = coords_3d.shape[:2]
        for i in range(h):
            for j in range(w):
                sphere = self.loader.loadModel('smiley')
                # texture = self.loader.loadTexture('maps/Dirlight.png')

                sphere.reparentTo(self.depth_node)

                if not use_relative_depths:
                    sphere.setScale(self.base_sphere_scale)
                else:
                    scale = self.base_sphere_scale + 0.1 * (relative_depths[i, j] - np.min(relative_depths))
                    sphere.setScale(scale)

                # sphere.setTexture(texture, 0)
                sphere.setPos(*coords_3d[i, j])
                sphere.setColor(*colors[i, j])

        return self.depth_node

    def _render_three_depth_maps(self):
        pass



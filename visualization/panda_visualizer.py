from direct.showbase.ShowBase import ShowBase
from direct.showbase.ShowBaseGlobal import globalClock
from direct.task import Task
from direct.showbase.Loader import Loader
from panda3d.core import *
import numpy as np
import cv2
import matplotlib as mpl
import matplotlib.cm as cm

from visualization.compute_3d_coordinates import compute_3d_coordinates

SIZE = 800


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


class Visualizer(ShowBase):

    def __init__(self, data):
        ShowBase.__init__(self)

        self.data = data

        self.fps = 30
        self.rotation_speed = 10
        self.current_pose_index = 0
        self.vertex_data = None
        self.configure()

    def configure(self):
        self.disableMouse()
        globalClock.setMode(ClockObject.MLimited)
        globalClock.setFrameRate(self.fps)
        self.create_window()
        self.configure_tasks()

    def create_window(self):
        wp = WindowProperties()
        wp.setTitle('3D depths')
        wp.setSize(SIZE, SIZE)
        wp.setOrigin(50, 50)

        self.winList[0].requestProperties(wp)
        self.camList[0].setPos(6, 6, 4)
        self.camList[0].lookAt(0, 0, 0)
        mk = self.dataRoot.attachNewNode(MouseAndKeyboard(self.winList[0], 0, 'w2mouse'))
        mk.attachNewNode(ButtonThrower('w2mouse'))

    def configure_tasks(self):
        self.taskMgr.add(self.auto_rotate, 'autoRotate')
        self.accept('q', self.finalizeExit)
        # self.configure_camera_controls()

    def configure_camera_controls(self):
        self.accept('w', self.move_forward)
        self.accept('s', self.move_backward)
        self.accept('a', self.move_left)
        self.accept('d', self.move_up)

    def next_step(self):
        self.current_pose_index += 1
        self.render_poses()

    def previous_step(self):
        self.current_pose_index -= 1
        self.render_poses()

    def auto_rotate(self, task):
        """
        Lets the camera rotate around the pose with speed self.rotation_speed
        :param task:
        :return:
        """
        time = task.time
        angle = time / self.fps * self.rotation_speed % 360
        for cam in self.camList:
            cam_position = np.asarray(cam.getPos())
            radius = np.linalg.norm(cam_position[:2])
            cam.setPos(np.cos(angle) * radius,
                       np.sin(angle) * radius,
                       cam_position[2])
            cam.lookAt(0, 0, 0)
        return Task.cont

    def compute_coloring(self, depths, downscale=1):
        h, w = depths.shape[:2]
        depths = cv2.resize(depths, (w // downscale, h // downscale))

        vmax = np.percentile(depths, 95)
        normalizer = mpl.colors.Normalize(vmin=depths.min(), vmax=vmax)
        mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
        colors = mapper.to_rgba(depths)[:, :, :3]
        return np.swapaxes(colors, 0, 1)

    def visualize_single_step(self, step_num=0):
        """
        Create simple animation of back projected 3D points without accounting for relative position change.
        """
        self.accept('+', self.next_step)
        self.accept('+-repeat', self.next_step)
        self.accept('-', self.previous_step)
        self.accept('--repeat', self.previous_step)

        downscale = 8
        predicted_depths = self.data['depth']
        coords = compute_3d_coordinates(self.data, downscale=downscale, global_coordinates=True)
        # swap z and y axis
        # coords = coords[..., [0, 2, 1]]
        # coords[..., 2] *= -1

        coords_3d = coords[step_num]

        # compute colors
        color_depths = predicted_depths[step_num]
        colors = self.compute_coloring(color_depths, downscale)

        root = NodePath('Depth 0')

        pose_node = self._render_single_depth_map(coords_3d, colors)

        axes_node = self.create_axes(2)
        axes_node.reparentTo(root)
        pose_node.reparentTo(root)
        self.camList[0].reparentTo(root)
        self.setBackgroundColor(200, 200, 200)

    def _render_single_depth_map(self, coords_3d, colors, center_of_projection=None):
        """
        Renders the given 3d coordinates as a point cloud.
        """
        node = NodePath('depth base node')

        h, w = coords_3d.shape[:2]
        for i in range(h):
            for j in range(w):
                smiley = self.loader.loadModel('smiley')
                smiley.reparentTo(node)
                smiley.setScale(0.01)
                smiley.setPos(*coords_3d[i, j])
                smiley.setColor(*colors[i, j])

        return node

    def create_axes(self, length=1):
        ls = LineSegs()
        ls.setThickness(1)

        # X axis
        ls.setColor(1.0, 0.0, 0.0, 1.0)
        ls.moveTo(0.0, 0.0, 0.0)
        ls.drawTo(1.0 * length, 0.0, 0.0)

        # Y axis
        ls.setColor(0.0, 1.0, 0.0, 1.0)
        ls.moveTo(0.0, 0.0, 0.0)
        ls.drawTo(0.0, 1.0 * length, 0.0)

        # Z axis
        ls.setColor(0.0, 0.0, 1.0, 1.0)
        ls.moveTo(0.0, 0.0, 0.0)
        ls.drawTo(0.0, 0.0, 1.0 * length)

        node = ls.create()
        return NodePath(node)

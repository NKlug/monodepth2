import cv2
import matplotlib as mpl
import matplotlib.cm as cm
import numpy as np
import tqdm
from direct.gui.OnscreenImage import OnscreenImage
from direct.task import Task
from panda3d.core import *

from visualization.compute_3d_coordinates import compute_3d_coordinates
from visualization.controllable_show_base import ControllableShowBase
from visualization.prepare_node import prepare_scatter_node


class Visualizer(ControllableShowBase):

    def __init__(self, data, precompute_nodes=True, render_mode='scatter', color_mode='depth', point_type='cube',
                 global_coordinates=True, downsample_factor=6, base_point_scale=0.005,
                 max_depth=1.5, use_relative_depths=False, show_2d_image=True, coords_upscale=1):
        """
        Creates the Panda3D Visualizer.
        @param data: data dictionary
        @param precompute_nodes: whether to precompute all nodes before visualizing. This means longer a longer loading
        time, but more fluent rendition.
        @param render_mode: 'scatter' or 'mesh'. Whether to render the points as a scatter plot or as a mesh.
        @param color_mode: 'depth' or 'image'. Whether choose the colors relative to the relative depths or to use
        the colors form the images.
        @param point_type: 'cube' or 'ball'. Whether to use a cube or a ball to represent a point in space
        @param global_coordinates: Whether to view the points in the global coordinate system.
        @param downsample_factor: Factor of how much to downsample the images.
        @param max_depth: Points with a relative depth greater than max_depth are not displayed.
        @param use_relative_depths: Whether to scale points relative to their depth.
        @param base_point_scale: Base size of a point.
        @param show_2d_image: Whether to show the 2d image from which the current 3d points originate.
        """
        ControllableShowBase.__init__(self)

        self._render_frame_map = {
            'scatter': self._render_frame_as_scatter,
            'mesh': self._render_frame_as_mesh
        }

        self.SINGLE_STEP = 0
        self.MULTI_STEP = 1
        self.EVERYTHING = 2

        self.data = data
        self.base_point_scale = base_point_scale

        self.render_mode = render_mode
        self.global_coordinates = global_coordinates
        self.max_depth = max_depth
        self.use_relative_depths = use_relative_depths
        self.show_2d_image = show_2d_image

        axes_node = self.create_axes_and_grid(length=20)
        axes_node.reparentTo(self.root)

        self.depth_node = None
        self.onscreenimage = None
        self.render_fn = None

        print('-> Preparing data')

        self.step_num = 0
        self.downsample = downsample_factor
        self.predicted_depths = self.data['depth']
        self.coords_3d, self.position, self.orientation = compute_3d_coordinates(self.data, downsample=self.downsample,
                                                                                 global_coordinates=self.global_coordinates)

        if not self.global_coordinates:
            self.coords_3d *= coords_upscale
            self.coords_3d -= np.min(self.coords_3d[...] - 1, axis=(0, 1, 2), keepdims=True)

        if point_type == 'cube':
            self.model_path = '../assets/cube.egg'
        elif point_type == 'ball':
            self.model_path = 'smiley'
        else:
            raise Exception(f"Unknown point type {point_type}!")

        if color_mode == 'depth':
            self.colors = np.asarray(
                [self.compute_depth_coloring(depths=d, downsample=self.downsample) for d in self.predicted_depths])
        elif color_mode == 'image':
            self.colors = np.asarray(
                [self.compute_image_coloring(image=img, downsample=self.downsample) for img in self.data['color']])
        else:
            raise Exception(f"Unknown coloring mode {color_mode}!")

        c, h, w = data['color'][0].shape
        self.dummy_image = np.zeros((h * 2, w * 2, c), dtype=np.uint8)
        self.texture_onscreenimage = Texture()
        self.texture_onscreenimage.setup2dTexture(w * 2, h * 2, Texture.T_unsigned_byte, Texture.F_rgb)

        self.nodes = [None] * len(self.coords_3d)
        self.render_single_fn = None
        if precompute_nodes:
            self._prepare_nodes()
            print("-> All nodes ready.")
            self.render_single_fn = self._show_single_depth_map
        else:
            self.render_single_fn = self._render_single_depth_map

    def compute_depth_coloring(self, depths, downsample=1, *args, **kwargs):
        h, w = depths.shape[:2]
        depths = cv2.resize(depths, (w // downsample, h // downsample))

        vmax = np.percentile(depths, 95)
        normalizer = mpl.colors.Normalize(vmin=depths.min(), vmax=vmax)
        mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
        colors = mapper.to_rgba(depths)[:, :, :3]
        return np.swapaxes(colors, 0, 1)

    def compute_image_coloring(self, image, downsample=1, *args, **kwargs):
        image = image.T
        h, w = image.shape[:2]
        return cv2.resize(image, (w // downsample, h // downsample))

    def visualize_with_animation(self, delay=200, step_num=0, *args, **kwargs):
        """
        Animates the 3d depth maps with automated camera movement.
        @param delay: delay between two frames in milliseconds
        @param step_num: starting step
        """
        self.step_num = step_num

        # prepare nodes if not available already
        if None in self.nodes:
            self._prepare_nodes()

        self.render_fn = self._show_single_depth_map

        self.camera.reparentTo(self.root)
        self.setBackgroundColor(200, 200, 200)

        self.yaw = self.orientation[self.step_num, 2] + 100
        self.pitch = np.maximum(self.orientation[self.step_num, 1] - 10, -90)
        self.camera.setHpr(self.yaw, self.pitch, 0)

        yaw = self.yaw / 180 * np.pi
        look_at_direction = np.asarray([-np.sin(yaw), np.cos(yaw), 0])
        look_at_direction = look_at_direction / np.linalg.norm(look_at_direction)

        right_direction = look_at_direction[[1, 0, 2]]
        right_direction[1] *= -1

        self.camera_position = self.position[self.step_num] + 0.3 * look_at_direction + \
                               -0.2 * right_direction + 1 * np.asarray([0, 0, 1.4])
        self.camera.setPos(*self.camera_position)

        self.taskMgr.doMethodLater(delay / 1000, self._animate_task, 'animateTask')

    def _animate_task(self, task, *args, **kwargs):
        self.step_num = (self.step_num + 1) % len(self.coords_3d)

        yaw = self.yaw / 180 * np.pi
        look_at_direction = np.asarray([-np.sin(yaw), np.cos(yaw), 0])
        look_at_direction = look_at_direction / np.linalg.norm(look_at_direction)

        right_direction = look_at_direction[[1, 0, 2]]
        right_direction[1] *= -1

        self.camera_position[:2] = (self.position[self.step_num] + 0.4 * look_at_direction + -0.2 * right_direction)[:2]
        self.camera.setPos(*self.camera_position)

        self._render(*args, **kwargs)

        return Task.again

    def _render(self, *args, **kwargs):
        self.depth_node.detachNode()
        self.depth_node = self.render_fn(*args, **kwargs)
        self.depth_node.reparentTo(self.root)

    def next_step(self, args, kwargs):
        self.step_num = (self.step_num + 1) % len(self.coords_3d)
        self._render(*args, **kwargs)

    def previous_step(self, args, kwargs):
        self.step_num = (self.step_num - 1) % len(self.coords_3d)
        self._render(*args, **kwargs)

    def visualize_with_steps(self, mode, step_num=0, interval_step=3, *args, **kwargs):
        """
        Create simple animation of back projected 3D points without accounting for relative position change.
        @param mode: visualization mode. One of Visualizer.SINGLE_STEP, Visualizer.MULTI_STEP, Visualizer.EVERYTHING.
        Displays either one single time step, three frames seperated by `interval_step` frames (with different opacities)
        or all frames simultaneously (which is somewhat slow).
        @param interval_step: Number of frames in between the three frames displayed in multi-step mode.
        @param step_num: Number of the frame to display.
        """
        self.step_num = step_num

        if mode == self.SINGLE_STEP:
            self.render_fn = self.render_single_fn
        elif mode == self.MULTI_STEP:
            self.render_fn = self._render_three_depth_maps
        elif mode == self.EVERYTHING:
            self.render_fn = self._render_everything
        else:
            raise Exception(f'Unknown mode {mode}!')

        self.accept('n', self.next_step, [args, kwargs])
        self.accept('n-repeat', self.next_step, [args, kwargs])
        self.accept('b', self.previous_step, [args, kwargs])
        self.accept('b-repeat', self.previous_step, [args, kwargs])
        self.addInstructions(0.80, "B N: previous and next frame.")

        self.depth_node = self.render_fn(*args, **kwargs)
        self.depth_node.reparentTo(self.root)

        self.camera.reparentTo(self.root)
        self.setBackgroundColor(200, 200, 200)  # swap z and y axis
        # coords = coords[..., [0, 2, 1]]
        # coords[..., 2] *= -1

    def _prepare_nodes(self, *args, **kwargs):
        old_step = self.step_num
        for i in tqdm.tqdm(range(len(self.coords_3d)), desc="-> Preparing nodes ", colour='white'):
            self.step_num = i
            self.nodes[i] = self._render_single_depth_map(*args, **kwargs)

        self.step_num = old_step

    def _show_single_depth_map(self, alpha=1.0, *args, **kwargs):
        node = self.nodes[self.step_num]
        node.setTransparency(True)
        node.setSa(alpha)
        return node

    def _render_single_depth_map(self, alpha=1.0, *args, **kwargs):
        """
        Renders the 3d coordinates at the current step as a point cloud.
        """
        if self.nodes[self.step_num] is not None:
            return self.nodes[self.step_num]

        coords_3d = self.coords_3d[self.step_num]
        # compute colors
        relative_depths = self.predicted_depths[self.step_num]
        colors = self.colors[self.step_num]

        # cv2.imshow('image', colors)
        # cv2.waitKey()

        self.depth_node = NodePath('depth base node')

        w, h = coords_3d.shape[:2]

        relative_depths = np.swapaxes(relative_depths, 0, 1)
        relative_depths = cv2.resize(relative_depths, (h, w))
        scale = -0.01 + 0.03 * relative_depths
        scale = np.maximum(scale, 0.005)
        scale = np.minimum(scale, 0.1)

        frame_node = self._render_frame_map[self.render_mode](alpha, colors, coords_3d, scale,
                                                              relative_depths,
                                                              *args,
                                                              **kwargs)
        frame_node.reparentTo(self.depth_node)

        if self.show_2d_image:
            self._show_2d_image()

        self.nodes[self.step_num] = self.depth_node
        return self.depth_node

    def _show_2d_image(self):
        if self.onscreenimage is not None:
            self.onscreenimage.removeNode()
        image = (self.data['color'][self.step_num].T * 255).astype(np.uint8)
        image = image[..., [2, 1, 0]]
        image = np.swapaxes(image, 0, 1)
        image = image[::-1, ...]
        h, w = image.shape[:2]
        self.dummy_image[h:, w:] = image

        self.texture_onscreenimage.setRamImage(self.dummy_image.tostring())
        s = 1

        self.onscreenimage = OnscreenImage(image=self.texture_onscreenimage)
        self.onscreenimage.setScale((s, 1., h / w * s))
        self.onscreenimage.setPos((0, 0, 0))
        self.onscreenimage.reparentTo(self.a2dBottomLeft)

    def _render_frame_as_mesh(self, alpha, colors, coords_3d, *args, **kwargs):
        w, h = coords_3d.shape[:2]
        ls = LineSegs()
        ls.setThickness(2)

        for i in range(w):
            for j in range(h):
                ls.setColor(*colors[i, j], alpha)

                if i < w - 1:
                    ls.move_to(*coords_3d[i, j])
                    ls.draw_to(*coords_3d[i + 1, j])

                if j < h - 1:
                    ls.move_to(*coords_3d[i, j])
                    ls.draw_to(*coords_3d[i, j + 1])

        return NodePath(ls.create())

    def _render_frame_as_scatter(self, alpha, colors, coords_3d, scale, relative_depths,
                                *args, **kwargs):
        return prepare_scatter_node(alpha, colors, coords_3d, scale, relative_depths, self.max_depth, self.loader,
                                    self.model_path, self.use_relative_depths, self.base_point_scale, *args, **kwargs)

    def _render_three_depth_maps(self, interval_step=1, *args, **kwargs):
        old_step_num = self.step_num
        indices = [self.step_num]
        if self.step_num - interval_step >= 0:
            indices.insert(0, self.step_num - interval_step)
        if self.step_num + interval_step < len(self.coords_3d):
            indices.append(self.step_num + interval_step)

        collector_node = NodePath('collector node')

        for j, i in enumerate(indices):
            self.step_num = i
            node = self.render_single_fn(alpha=(j + 1) / (len(indices) + 1), *args, **kwargs)
            node.reparentTo(collector_node)

        self.step_num = old_step_num
        return collector_node

    def _render_everything(self, *args, **kwargs):
        collector_node = NodePath('collector node')

        for i in range(len(self.coords_3d)):
            self.step_num = i
            node = self.render_single_fn(*args, **kwargs)
            node.reparentTo(collector_node)

        return collector_node

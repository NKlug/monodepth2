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
import tqdm

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


class Visualizer(ControllableShowBase):

    def __init__(self, data, precompute_nodes=True, render_mode='scatter'):
        ControllableShowBase.__init__(self)

        self._render_frame_map = {
            'scatter': self._render_frame_as_scatter,
            'mesh': self._render_frame_as_mesh
        }

        self.SINGLE_STEP = 0
        self.MULTI_STEP = 1

        self.data = data
        self.base_sphere_scale = 0.01

        self.render_mode = render_mode

        axes_node = self.create_axes_and_grid(length=20)
        axes_node.reparentTo(self.root)

        self.depth_node = None
        self.render_fn = None

        print('-> Preparing data')

        self.step_num = 0
        self.downscale = 8
        self.predicted_depths = self.data['depth']
        self.coords_3d = compute_3d_coordinates(self.data, downscale=self.downscale, global_coordinates=True)
        self.colors = np.asarray([self.compute_coloring(d, self.downscale) for d in self.predicted_depths])

        self.nodes = [None] * len(self.coords_3d)
        self.render_single_fn = None
        if precompute_nodes:
            self._prepare_nodes()
            print("-> All nodes ready.")
            self.render_single_fn = self._show_single_depth_map
        else:
            self.render_single_fn = self._render_single_depth_map

    def compute_coloring(self, depths, downscale=1):
        h, w = depths.shape[:2]
        depths = cv2.resize(depths, (w // downscale, h // downscale))

        vmax = np.percentile(depths, 95)
        normalizer = mpl.colors.Normalize(vmin=depths.min(), vmax=vmax)
        mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
        colors = mapper.to_rgba(depths)[:, :, :3]
        return np.swapaxes(colors, 0, 1)

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

    def visualize_with_steps(self, mode, step_num=0, *args, **kwargs):
        """
        Create simple animation of back projected 3D points without accounting for relative position change.
        """
        self.step_num = step_num

        if mode == self.SINGLE_STEP:
            self.render_fn = self.render_single_fn
        elif mode == self.MULTI_STEP:
            self.render_fn = self._render_three_depth_maps
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

    def _render_single_depth_map(self, use_relative_depths=True, alpha=1.0, *args, **kwargs):
        """
        Renders the 3d coordinates at the current step as a point cloud.
        """
        if self.nodes[self.step_num] is not None:
            return self.nodes[self.step_num]

        coords_3d = self.coords_3d[self.step_num]
        # compute colors
        relative_depths = self.predicted_depths[self.step_num]
        colors = self.colors[self.step_num]

        self.depth_node = NodePath('depth base node')

        w, h = coords_3d.shape[:2]

        relative_depths = np.swapaxes(relative_depths, 0, 1)
        relative_depths = cv2.resize(relative_depths, (h, w))
        scale = -0.01 + 0.03 * relative_depths
        scale = np.maximum(scale, 0.005)
        scale = np.minimum(scale, 0.1)

        frame_node = self._render_frame_map[self.render_mode](alpha, colors, coords_3d, scale, use_relative_depths,
                                                              *args,
                                                              **kwargs)
        frame_node.reparentTo(self.depth_node)

        self.nodes[self.step_num] = self.depth_node
        return self.depth_node

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

    def _render_frame_as_scatter(self, alpha, colors, coords_3d, scale, use_relative_depths):
        w, h = coords_3d.shape[:2]
        frame_node = NodePath('frame node')

        for i in range(w):
            for j in range(h):
                sphere = self.loader.loadModel('smiley')
                texture = self.loader.loadTexture('../assets/sphere.rgb')

                sphere.reparentTo(frame_node)

                if not use_relative_depths:
                    sphere.setScale(self.base_sphere_scale)
                else:
                    sphere.setScale(scale[i, j])
                    # sphere.setScale(np.maximum(0.01, np.random.normal(0.03, 0.01)))

                sphere.setTexture(texture, 1)
                sphere.setPos(*coords_3d[i, j])
                sphere.setTransparency(True)
                sphere.setColor(*colors[i, j], alpha)

        return frame_node

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

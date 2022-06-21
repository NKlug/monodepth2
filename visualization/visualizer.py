import cv2
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as cm
import scipy.spatial as spat

from visualization.compute_3d_coordinates import get_global_coords, compute_3d_coordinates


class Visualizer:

    def __init__(self, data):
        self.data = data
        self.fig = plt.figure(figsize=(20, 10))
        self.fig.tight_layout()
        self.ax = self.fig.add_subplot(111, projection='3d')

    def plot_camera_path(self):

        lat, lon, alt, *_ = get_global_coords(self.data)
        self.ax.plot(lat, lon, alt)

    def compute_coloring(self, depths):
        vmax = np.percentile(depths, 95)
        normalizer = mpl.colors.Normalize(vmin=depths.min(), vmax=vmax)
        mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
        return mapper.to_rgba(depths)[:, :, :3]

    def _plot_camera(self, position, orientation, scale=1):
        """
        Plots a camera model at the given position oriented towards the given orientation vector (roll, pitch, yaw)
        """
        lu = [0, 1, 3]
        ru = [4, 1, 3]
        lb = [0, 1, 0]
        rb = [4, 1, 0]
        c = [2, 0, 1.5]
        corners = np.stack([lu, ru, lb, rb, c], axis=0)
        # put center of image plane at (0, 0) and normalize
        corners = (corners - np.asarray([2, 1, 1.5])) / 4

        rot = spat.transform.Rotation.from_euler('xyz', orientation).as_matrix()

        corners = scale * (rot @ corners.T).T + np.repeat(position[np.newaxis, :], 5, axis=0)

        lu, ru, lb, rb, c = corners
        lines_to_plot = [[lu, lb], [lb, rb], [rb, ru], [ru, lu], [lu, c], [ru, c], [lb, c], [rb, c]]
        lines_to_plot = np.asarray(lines_to_plot)
        for line in lines_to_plot:
            self.ax.plot(line[:, 0], line[:, 1], line[:, 2], color='black')

    def visualize_with_steps(self):
        """
        Visualizes data in single steps.
        Press <space> for forward, < - > for backward.
        """
        downscale = 8

        # compute and extract some data
        predicted_depths = self.data['depth']
        coords = compute_3d_coordinates(self.data, downscale=downscale, global_coordinates=True)
        lat, lon, alt, roll, pitch, yaw = get_global_coords(self.data)

        def on_press(event):
            if event is not None:
                if event.key == ' ':
                    on_press.num = (on_press.num + 1) % len(coords)
                elif event.key == '-':
                    on_press.num = (on_press.num - 1) % len(coords)

            self.ax.clear()

            i = on_press.num

            # plot camera path
            # self.plot_camera_path()

            # plot camera
            position = np.asarray([lat[i], lon[i], alt[i]])
            orientation = np.asarray([roll[i], pitch[i], yaw[i] - np.pi / 2])
            self._plot_camera(position, orientation)

            # add bouding points for equal axes scales
            MAX = 5
            for direction in (-1, 1):
                for point in np.diag(direction * MAX * np.array([1, 1, 1])):
                    self.ax.plot([point[0]], [point[1]], [point[2]], 'green')

            # extract coordinates
            xv = coords[on_press.num, :, :, 0]
            yv = coords[on_press.num, :, :, 1]
            zv = coords[on_press.num, :, :, 2]

            # compute colors
            color_depths = predicted_depths[on_press.num]
            h, w = color_depths.shape[:2]
            color_depths = cv2.resize(color_depths, (w // downscale, h // downscale))
            colors = self.compute_coloring(color_depths)

            self.ax.set_xlabel('X')
            self.ax.set_ylabel('Y')
            self.ax.set_zlabel('Z')
            self.ax.scatter(xv.ravel(), yv.ravel(), zv.ravel(), s=200, c=colors.reshape((-1, 3)))
            self.fig.canvas.draw()

        on_press.num = 0
        on_press(None)
        self.fig.canvas.mpl_connect('key_press_event', on_press)
        plt.show()

    def simple_visualize_sequence(self):
        """
        Create simple animation of back projected 3D points without accounting for relative position change.
        """
        downscale = 8
        predicted_depths = self.data['depth']
        coords = compute_3d_coordinates(self.data, downscale=downscale)

        def next_scatter(num, coords):
            xv = coords[num, :, :, 0]
            yv = coords[num, :, :, 1]
            zv = coords[num, :, :, 2]

            # compute colors
            color_depths = predicted_depths[num]
            h, w = color_depths.shape[:2]
            color_depths = cv2.resize(color_depths, (w // downscale, h // downscale))
            colors = self.compute_coloring(color_depths)

            # plot data
            self.ax.clear()
            self.ax.set_xlabel('X')
            self.ax.set_ylabel('Y')
            self.ax.set_zlabel('Z')

            # add bouding points for equal axes scales
            MAX = 1
            for direction in (-1, 1):
                for point in np.diag(direction * MAX * np.array([1, 1, 1])):
                    self.ax.plot([point[0]], [point[1]], [point[2]], 'green')

            scattered = self.ax.scatter(xv.ravel(), yv.ravel(), zv.ravel(), s=200, c=colors.reshape((-1, 3)))

        # create animation
        ani = animation.FuncAnimation(self.fig, next_scatter, len(coords), fargs=(coords,), interval=100, blit=False)

        def on_press(event):
            if event.key == ' ':
                if on_press.pause:
                    ani.event_source.start()
                else:
                    ani.event_source.stop()
                on_press.pause = not on_press.pause

        on_press.pause = False
        self.fig.canvas.mpl_connect('key_press_event', on_press)

        plt.show()

    def visualize_single_step(self, step_num):
        """
        Create simple animation of back projected 3D points without accounting for relative position change.
        """
        downscale = 8
        predicted_depths = self.data['depth']
        coords = compute_3d_coordinates(self.data, downscale=downscale)

        xv = coords[step_num, :, :, 0]
        yv = coords[step_num, :, :, 1]
        zv = coords[step_num, :, :, 2]

        # compute colors
        color_depths = predicted_depths[step_num]
        h, w = color_depths.shape[:2]
        color_depths = cv2.resize(color_depths, (w // downscale, h // downscale))
        colors = self.compute_coloring(color_depths)
        colors = np.swapaxes(colors, 0, 1)

        # add bouding points for equal axes scales
        MAX = 1
        for direction in (-1, 1):
            for point in np.diag(direction * MAX * np.array([1, 1, 1])):
                self.ax.plot([point[0]], [point[1]], [point[2]], 'green')

        # plot data
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        scattered = self.ax.scatter(xv.ravel(), yv.ravel(), zv.ravel(), s=200, c=colors.reshape((-1, 3)))

        plt.show()
        cv2.waitKey()

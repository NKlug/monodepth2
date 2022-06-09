import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import matplotlib.cm as cm
import scipy.spatial as spat

from utils import lat_lon_to_meters
from visualization.visualize_sequence import compute_3d_coordinates


class Visualizer:

    def __init__(self, data):
        self.data = data
        self.fig = plt.figure(figsize=(20, 10))
        self.fig.tight_layout()
        self.ax = self.fig.add_subplot(111, projection='3d', proj_type='persp')

    def _get_global_coords(self):
        lat = self.data['oxts']['lat']
        lon = self.data['oxts']['lon']
        alt = self.data['oxts']['alt']

        roll = self.data['oxts']['roll']
        pitch = self.data['oxts']['pitch']
        yaw = self.data['oxts']['yaw']

        lat, lon = lat_lon_to_meters(lat, lon)
        lat = lat - lat[0]
        lon = lon - lon[0]
        alt = alt - alt[0] + 1

        return lat, lon, alt, roll, pitch, yaw

    def plot_camera_path(self):

        lat, lon, alt, *_ = self._get_global_coords()

        self.ax.plot(lat, lon, alt)

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
        coords = self._compute_3d_coordinates(global_coordinates=True)

        lat, lon, alt, roll, pitch, yaw = self._get_global_coords()

        self.plot_camera_path()

        def on_press(event):
            if event is not None:
                if event.key == ' ':
                    on_press.num = (on_press.num + 1) % len(coords)
                elif event.key == '-':
                    on_press.num = (on_press.num - 1) % len(coords)

            # self.ax.clear()

            i = on_press.num

            # plot camera
            position = np.asarray([lat[i], lon[i], alt[i]])
            orientation = np.asarray([roll[i], pitch[i], yaw[i] - np.pi / 2])

            self._plot_camera(position, orientation)

            MAX = 50
            for direction in (-1, 1):
                for point in np.diag(direction * MAX * np.array([1, 1, 1])):
                    self.ax.plot([point[0]], [point[1]], [point[2]], 'green')

            # plot 3d cloud
            xv = coords[on_press.num, :, :, 0]
            yv = coords[on_press.num, :, :, 1]
            zv = coords[on_press.num, :, :, 2]
            vmax = np.percentile(zv, 95)
            normalizer = mpl.colors.Normalize(vmin=zv.min(), vmax=vmax)
            mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
            colormapped_im = mapper.to_rgba(zv)[:, :, :3]

            self.ax.clear()
            self.ax.set_xlabel('X')
            self.ax.set_ylabel('Y')
            self.ax.set_zlabel('Z')
            self.ax.scatter(xv.flatten(), yv.flatten(), zv.flatten(), s=200, c=colormapped_im.reshape((-1, 3)))
            self.fig.canvas.draw()

        on_press.num = 0
        on_press(None)
        self.fig.canvas.mpl_connect('key_press_event', on_press)
        plt.show()

    def _compute_3d_coordinates(self, global_coordinates=False):
        """
        Compute 3d coordinates from predicted depth and camera intrinsics in meters.
        """
        predicted_depths = self.data["disp"]

        lat, lon, alt, roll, pitch, yaw = self._get_global_coords()

        coords = []
        for i, predicted_depth in enumerate(predicted_depths):
            coords_3d = compute_3d_coordinates(predicted_depth, in_imu_coordinates=global_coordinates,
                                               interim_downscale=8)

            if global_coordinates:
                # compute coordinates in global coordinate system
                position = np.asarray([lat[i], lon[i], alt[i]])
                orientation = np.asarray([roll[i], pitch[i], yaw[i] - np.pi / 2])
                rot = spat.transform.Rotation.from_euler('xyz', orientation).as_matrix()

                global2imu = np.eye(4)
                global2imu[:3, :3] = rot
                global2imu[3, :3] = position
                imu2global = np.linalg.inv(global2imu)

                h, w = coords_3d.shape[:2]
                coords_3d = np.concatenate([coords_3d, np.ones((h, w, 1))], axis=-1)
                coords_3d = coords_3d.reshape((-1, 4))

                coords_3d = np.dot(imu2global, coords_3d.T).T

                coords_3d = coords_3d[:, :3]
                coords_3d = coords_3d.reshape((h, w, 3))

            coords.append(coords_3d)

        return np.asarray(coords)

    def simple_visualize_sequence(self):
        coords = self._compute_3d_coordinates()

        fig = plt.figure(figsize=(20, 10))
        fig.tight_layout()
        ax = fig.add_subplot(111, projection='3d', proj_type='persp')

        def next_scatter(num, coords):
            xv = coords[num, :, :, 0]
            yv = coords[num, :, :, 1]
            zv = coords[num, :, :, 2]
            vmax = np.percentile(zv, 95)
            normalizer = mpl.colors.Normalize(vmin=zv.min(), vmax=vmax)
            mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
            colormapped_im = mapper.to_rgba(zv)[:, :, :3]
            ax.clear()
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            scattered = ax.scatter(xv.flatten(), yv.flatten(), zv.flatten(), s=200, c=colormapped_im.reshape((-1, 3)))
            # scattered = ax.plot_surface(xv, yv, zv, cmap='coolwarm', linewidth=1, antialiased=True)
            # MAX = 5
            # for direction in (-1, 1):
            #     for point in np.diag(direction * MAX * np.array([1, 1, 1])):
            #         ax.plot([point[0]], [point[1]], [point[2]], 'w')

            # scattered = ax.scatter(xv.flatten(), yv.flatten(), zv.flatten(), s=2)
            # for line, data in zip(lines, dataLines):
            #     # NOTE: there is no .set_data() for 3 dim data...
            #     line.set_data(data[0:2, :num])
            #     line.set_3d_properties(data[2, :num])
            # scatter_plots[num-1].set_visible(False)
            # scatter_plots[num].set_visible(True)

        line_ani = animation.FuncAnimation(fig, next_scatter, len(coords), fargs=(coords,), interval=100, blit=False)
        plt.show()

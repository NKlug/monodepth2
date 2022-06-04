import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import matplotlib.cm as cm
import scipy.spatial as spat

from visualization.visualize_sequence import compute_3d_coordinates


class Visualizer:

    def __init__(self, data):
        self.data = data
        self.fig = plt.figure(figsize=(20, 10))
        self.fig.tight_layout()
        self.ax = self.fig.add_subplot(111, projection='3d', proj_type='persp')

    def visualize_camera_path(self):

        lat = self.data['oxts']['lat']
        lon = self.data['oxts']['lon']
        alt = self.data['oxts']['alt']
        lat = lat - lat[0]
        lon = lon - lon[0]
        alt = alt - alt[0]

        self.ax.plot(lat, lon, 0 * alt + 1)
        plt.show()

    def _plot_camera(self, position, orientation, scale=1):
        """
        Plots a camera model at the given position oriented towards the given orientation vector (roll, pitch, yaw)
        """
        lu = [0, 1, 1]
        ru = [4, 1, 1]
        lb = [0, 1, 0]
        rb = [4, 1, 0]
        c = [2, 0, 2]
        corners = np.stack([lu, ru, lb, rb, c], axis=0)

        rot = spat.transform.Rotation.from_euler('xyz', orientation).as_matrix()

        corners = scale * (rot @ corners.T).T + np.repeat(position[np.newaxis, :], 5, axis = 0)

        self.ax.plot([lu, lb])
        self.ax.plot([lb, rb])
        self.ax.plot([rb, ru])
        self.ax.plot([ru, lu])

        self.ax.plot([])


    def visualize_with_steps(self):
        """
        Visualizes data in single steps.
        Press <space> for forward, < - > for backward.
        """
        coords = self._compute_3d_coordinates()

        def on_press(event):
            if event is not None:
                if event.key == ' ':
                    on_press.num = (on_press.num + 1) % len(coords)
                elif event.key == '-':
                    on_press.num = (on_press.num - 1) % len(coords)

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

    def _compute_3d_coordinates(self):
        predicted_depths = self.data["disp"]

        coords = []
        for predicted_depth in predicted_depths:
            coords_3d = compute_3d_coordinates(predicted_depth, interim_downscale=8)
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

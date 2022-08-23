import pickle
import os
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np

from options import MonodepthOptions


def compute_coloring(depths):
    vmax = np.percentile(depths, 95)
    normalizer = mpl.colors.Normalize(vmin=depths.min(), vmax=vmax)
    mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
    colors = mapper.to_rgba(depths)[:, :, :3]
    return colors


def create_2d_animation(data):
    depths = data['depth']
    images = data['color']

    fig = plt.figure(figsize=(20, 10))
    ax1 = plt.subplot(1, 2, 1)
    ax2 = plt.subplot(1, 2, 2)

    def animate(i):
        ax1.clear()
        ax2.clear()
        color_image = np.swapaxes(images[i].T, 0, 1)
        ax1.imshow(color_image)
        colored_depth = compute_coloring(depths[i])
        ax2.imshow(colored_depth)
        return ax1, ax2

    # animate(4)
    anim = animation.FuncAnimation(fig, animate,
                                   frames=len(depths), interval=15, blit=True)
    plt.show()

    # print('-> Saving...')
    # plt.rcParams['savefig.bbox'] = 'tight'
    # anim.save('basic_animation.mp4', writer='ffmpeg', fps=30, extra_args=['-vcodec', 'libx264'])


if __name__ == '__main__':
    split = None
    split = 'Mannequin_00c9878266685887_custom'
    # split = 'Mannequin_ebdf4f0a220e58e6_custom'

    load_weights_folder = None
    load_weights_folder = '/home/nikolas/Projects/monodepth2/models/mannequin/models/weights_4'
    load_weights_folder = '/home/nikolas/Projects/monodepth2/models/mono_model/models/weights_19'

    options = MonodepthOptions()
    opt = options.parse()
    if split is not None:
        opt.split = split
    if load_weights_folder is not None:
        opt.load_weights_folder = load_weights_folder

    # load predicted depths
    data_path = os.path.join(
        opt.load_weights_folder, "predicted_depths_{}_split.pkl".format(opt.split))
    if not os.path.exists(data_path):
        print("Depth predictions not found. Transfer from server first!")
        exit(0)
    print("-> Loading predicted depths from ", data_path)
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    create_2d_animation(data)

import os
import pickle
import numpy as np

from options import MonodepthOptions
from visualization.visualizer import Visualizer as MatplotVisualizer
from visualization.panda_visualizer import Visualizer as PandaVisualizer

if __name__ == '__main__':
    split = '2011_09_30_drive_0020'
    split = '2011_09_28_drive_0002'
    split = '2011_09_26_drive_0001'
    split = None
    options = MonodepthOptions()
    opt = options.parse()

    if split is not None:
        opt.split = split

    data_path = os.path.join(
        opt.load_weights_folder, "predicted_depths_{}_split.pkl".format(opt.split))
    if not os.path.exists(data_path):
        print("Depth predictions not found. Transfer from server first!")
        exit(0)

    print("-> Loading predicted depths from ", data_path)
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    app = PandaVisualizer(data, precompute_nodes=False, render_mode='scatter', color_mode='image', point_type='ball',
                          global_coordinates=True, max_depth=1.5, use_relative_depths=False, downscale_factor=5)
    app.visualize_with_steps(app.SINGLE_STEP, step_num=1, interval_step=5)
    # app.visualize_with_animation()
    app.run()

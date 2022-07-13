import os
import pickle
import numpy as np

from options import MonodepthOptions
from visualization.visualizer import Visualizer as MatplotVisualizer
from visualization.panda_visualizer import Visualizer as PandaVisualizer

if __name__ == '__main__':
    split = '2011_09_28_drive_0002'
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

    # visualizer = MatplotVisualizer(data)
    # visualizer.visualize_single_step(0)
    # visualizer.simple_visualize_sequence()
    # visualizer.visualize_with_steps()

    app = PandaVisualizer(data, precompute_nodes=False, render_mode='scatter', global_coordinates=False)
    app.visualize_with_steps(app.MULTI_STEP, step_num=1, interval_step=1, use_relative_depths=False)
    # app.visualize_with_animation()
    app.run()

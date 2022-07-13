import os
import pickle
import numpy as np

from options import MonodepthOptions
from visualization.visualizer import Visualizer as MatplotVisualizer
from visualization.panda_visualizer import Visualizer as PandaVisualizer

if __name__ == '__main__':
    if __name__ == '__main__':
        options = MonodepthOptions()
        opt = options.parse()

        output_path = os.path.join(
            opt.load_weights_folder, "predicted_depths_{}_split.pkl".format(opt.eval_split))
        print("-> Loading predicted depths from ", output_path)
        with open(output_path, 'rb') as f:
            data = pickle.load(f)

        # visualizer = MatplotVisualizer(data)
        # visualizer.visualize_single_step(0)
        # visualizer.simple_visualize_sequence()
        # visualizer.visualize_with_steps()

        app = PandaVisualizer(data, precompute_nodes=False, render_mode='scatter')
        app.visualize_with_steps(app.MULTI_STEP, step_num=1, interval_step=1)
        app.run()

import os
import pickle
import numpy as np

from options import MonodepthOptions
from visualization.visualizer import Visualizer

if __name__ == '__main__':
    if __name__ == '__main__':
        options = MonodepthOptions()
        opt = options.parse()

        output_path = os.path.join(
            opt.load_weights_folder, "predicted_depths_{}_split.pkl".format(opt.eval_split))
        print("-> Loading predicted depths from ", output_path)
        with open(output_path, 'rb') as f:
            data = pickle.load(f)

        visualizer = Visualizer(data)
        # visualizer.visualize_single_step(0)
        visualizer.simple_visualize_sequence()
        # visualizer.visualize_with_steps()

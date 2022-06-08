from options import MonodepthOptions
import os
import pickle
import numpy as np

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

        camera_coordinates = [(0, 0, 1, 0, 0, 0)]

        # data = {"disp": np.load('/home/nikolas/Projects/monodepth2/assets/test_image_disp.npy')[:, 0]}

        # data = back_project_depths(data, opt)
        visualizer = Visualizer(data)
        # visualizer.simple_visualize_sequence()
        visualizer.visualize_with_steps()
        # visualizer.visualize_camera_path()
        # visualizer._plot_camera(np.asarray([0, 0, 0]), [0, 0, 0])

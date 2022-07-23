import os
import pickle

from options import MonodepthOptions
from visualization.panda_visualizer import Visualizer as PandaVisualizer

if __name__ == '__main__':
    split = '2011_09_30_drive_0020'
    split = '2011_09_28_drive_0002'
    split = '2011_09_26_drive_0017'
    split = '2011_09_29_drive_0026'
    split = None  # use split passed through cli options

    # parse CLI options
    options = MonodepthOptions()
    opt = options.parse()
    if split is not None:
        opt.split = split

    # load predicted depths
    data_path = os.path.join(
        opt.load_weights_folder, "predicted_depths_{}_split.pkl".format(opt.split))
    if not os.path.exists(data_path):
        print("Depth predictions not found. Transfer from server first!")
        exit(0)
    print("-> Loading predicted depths from ", data_path)
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    # create visualizer with desired options
    app = PandaVisualizer(data, precompute_nodes=False, render_mode='scatter', color_mode='image', point_type='ball',
                          global_coordinates=True, max_depth=1.5, use_relative_depths=False, downscale_factor=5,
                          show_2d_image=True)

    # visualize with steps
    app.visualize_with_steps(app.SINGLE_STEP, step_num=0, interval_step=5)

    # app.visualize_with_animation()
    app.run()

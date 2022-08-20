from options import MonodepthOptions
import os
import numpy as np
from tqdm import tqdm


def determine_average_intrinsics_mannequin(split_filename, opt):
    intrinsics_list = []
    with open(split_filename, 'r') as split_file:
        frames = split_file.readlines()

    frames = sorted(frames)

    for frame in tqdm(frames):
        video, frame_num, _ = frame.split()
        intrinsics_file_name = video + '.txt'
        intrinsics_file_name = os.path.join(opt.data_path, intrinsics_file_name)

        # we assume the intrinsics stay the same throughout the video. Hence we use the intrinsics from the first frame
        with open(intrinsics_file_name, 'r') as intrinsics_file:
            _ = intrinsics_file.readline()  # skip first line (video url)
            line = intrinsics_file.readline()
            intrinsics = line.split()[1:7]

        # for the meaning of the values see https://google.github.io/mannequinchallenge/www/download.html
        K = np.zeros((4, 4), dtype=np.float32)
        K[0, 0] = intrinsics[0]
        K[1, 1] = intrinsics[1]
        K[0, 2] = intrinsics[2]
        K[1, 2] = intrinsics[3]
        K[2, 2] = 1
        K[3, 3] = 1
        intrinsics_list.append(K)

    intrinsics_list = np.asarray(intrinsics_list, dtype=np.float32)
    print(np.mean(intrinsics_list, axis=0))

if __name__ == '__main__':
    opt = MonodepthOptions().parse()
    folder = 'train'
    split_filename = 'mannequin_train'

    split_filename = os.path.join(split_filename, 'train_files.txt')
    determine_average_intrinsics_mannequin(split_filename, opt)

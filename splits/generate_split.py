import os
import glob

import cv2
import numpy as np

from options import MonodepthOptions

side_map = {"l": 2, "r": 3}


def generate_sequence_split_kitti(split_filename, folder, side, opt):
    image_directory = os.path.join(opt.data_path, folder, "image_0{}/data/*".format(side_map[side]))
    os.makedirs(os.path.dirname(split_filename), exist_ok=True)
    with open(os.path.join(split_filename), 'w') as split_file:
        for image_name in glob.glob(image_directory):
            relative_image_path = folder.replace(opt.data_path, '')
            if relative_image_path.startswith('/'):
                relative_image_path = relative_image_path[1:]
            index = os.path.basename(image_name).split('.')[0]
            split_file.write(f'{relative_image_path} {index} {side}\n')


def generate_video_split_mannequin(split_filename, video_name, folder,  opt, all_frames=False):
    video_path = os.path.join(opt.data_path, folder, video_name + '.mp4')
    annotations_path = os.path.join(opt.data_path, folder, video_name + '.txt')
    os.makedirs(os.path.dirname(split_filename), exist_ok=True)
    vidcap = cv2.VideoCapture(video_path)
    fps = vidcap.get(cv2.CAP_PROP_FPS)

    with open(annotations_path, 'r') as annotations_file:
        annotations = annotations_file.readlines()
        annotations = annotations[1:]  # discard first line (its the url)

    timestamps = np.array([int(line.split()[0]) for line in annotations])
    timestamps = timestamps / 1e6
    frame_numbers = np.round(fps * timestamps).astype(np.int)

    with open(split_filename, 'w') as split_file:
        for frame_number in frame_numbers:
            split_file.write(f'{video_name} {frame_number} 0\n')


if __name__ == '__main__':
    options = MonodepthOptions().parse()

    # code for kitti
    # side = 'l'
    # for folder in glob.glob(os.path.join(options.data_path, '*')):
    #     folder = os.path.basename(folder)
    #     for drive in glob.glob(os.path.join(options.data_path, folder, "*drive*")):
    #         split_name = os.path.basename(drive).replace('_sync', '')
    #         split_filename = os.path.join(split_name, 'test_files.txt')
    #         print(f'Generating split {split_filename} from folder {drive}')
    #         generate_split_kitti(split_filename, drive, side, options)

    # code for mannequin
    folder = 'validation'
    split_filename = '00c9878266685887'
    video_name = split_filename
    split_filename = f'Mannequin_{split_filename}'

    split_filename = os.path.join(split_filename, 'test_files.txt')
    print(f'Generating split {split_filename} from video {video_name}')
    generate_video_split_mannequin(split_filename, video_name, folder, options)


    # folder = '2011_09_26/2011_09_26_drive_0001_sync'
    # split_filename = 'sequence/test_files.txt'
    # generate_split_file(split_filename, folder, side, options)

import os
import glob

from options import MonodepthOptions

side_map = {"l": 2, "r": 3}


def generate_split_file(split_filename, folder, side, opt):
    image_directory = os.path.join(opt.data_path, folder, "image_0{}/data/*".format(side_map[side]))
    os.makedirs(os.path.dirname(split_filename), exist_ok=True)
    with open(os.path.join(split_filename), 'w') as split_file:
        for image_name in glob.glob(image_directory):
            index = os.path.basename(image_name).split('.')[0]
            split_file.write(f'{folder} {index} {side}\n')


if __name__ == '__main__':
    options = MonodepthOptions().parse()
    side = 'l'
    for folder in glob.glob(os.path.join(options.data_path, '*')):
        folder = os.path.basename(folder)
        for drive in glob.glob(os.path.join(options.data_path, folder, "*drive*")):
            split_name = os.path.basename(drive).replace('_sync', '')
            split_filename = os.path.join(split_name, 'test_files.txt')
            print(f'Generating split {split_filename} from folder {drive}')
            generate_split_file(split_filename, drive, side, options)

    # folder = '2011_09_26/2011_09_26_drive_0001_sync'
    # split_filename = 'sequence/test_files.txt'
    # generate_split_file(split_filename, folder, side, options)

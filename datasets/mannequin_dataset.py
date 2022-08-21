import os
import numpy as np
from datasets.mono_dataset import MonoDataset
import PIL.Image as pil

import imageio
import cv2


def mp4_loader(path, frame_id):
    """
    MP4 loader.
    @param frame_id: frame id
    @param path: Path of the mp4 file
    @return:
    """
    if not os.path.exists(path):
        raise Exception(f'Video file at {path} does not exist!')
    video = imageio.get_reader(path, 'ffmpeg')
    image = video.get_data(frame_id)
    if image is None:
        raise Exception(f'Could not read frame {frame_id} from {path}!')
    return pil.fromarray(image)


class MannequinDataset(MonoDataset):
    """Superclass for different types of MannequinDataset loaders
    """

    def __init__(self, *args, **kwargs):
        super(MannequinDataset, self).__init__(*args, **kwargs)

        self.loader = mp4_loader

        # NOTE: Make sure your intrinsics matrix is *normalized* by the original image size.
        # To normalize you need to scale the first row by 1 / image_width and the second row
        # by 1 / image_height. Monodepth2 assumes a principal point to be exactly centered.
        # If your principal point is far from the center you might need to disable the horizontal
        # flip augmentation.
        # self.K = np.array([[0.58, 0, 0.5, 0],
        #                    [0, 1.92, 0.5, 0],
        #                    [0, 0, 1, 0],
        #                    [0, 0, 0, 1]], dtype=np.float32)

        self.K = None

    def check_depth(self):
        return False

    def check_oxts(self):
        return False

    def get_camera_intrinsics(self):
        raise NotImplementedError()


class MultiVideoMannequinDataset(MannequinDataset):
    """General dataset class for the the MannequinDataset.
    """

    def __init__(self, *args, **kwargs):
        super(MultiVideoMannequinDataset, self).__init__(*args, **kwargs)

        # We use the average intrinsics of each video

        # NOTE: Make sure your intrinsics matrix is *normalized* by the original image size.
        # To normalize you need to scale the first row by 1 / image_width and the second row
        # by 1 / image_height. Monodepth2 assumes a principal point to be exactly centered.
        # If your principal point is far from the center you might need to disable the horizontal
        # flip augmentation.
        self.K = np.array([[0.88, 0, 0.5, 0],
                           [0, 1.58, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)

    def get_color(self, video_name, frame_index, side, do_flip):
        video_path = os.path.join(self.data_path, video_name + '.mp4')
        color_image = self.loader(video_path, frame_index)

        # resize without keeping the aspect ratio
        color_image.resize((self.width, self.height))

        if do_flip:
            color_image = color_image.transpose(pil.FLIP_LEFT_RIGHT)

        return color_image


class SingleVideoMannequinDataset(MannequinDataset):
    """Class for a single video of the MannequinDataset, i.e. all frames in the dataset stem from the same video.
    Uses the respective camera intrinsics
    """

    def __init__(self, *args, **kwargs):
        super(SingleVideoMannequinDataset, self).__init__(*args, **kwargs)

        self.K = self.get_camera_intrinsics()

    def get_camera_intrinsics(self):
        # all filenames are the same, as the dataset consists of only frames from one video.
        intrinsics_file_name = self.filenames[0].split()[0] + '.txt'
        intrinsics_file_name = os.path.join(self.data_path, intrinsics_file_name)

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

        return K

    def get_color(self, video_name, frame_index, side, do_flip):
        video_path = os.path.join(self.data_path, video_name + '.mp4')
        color_image = self.loader(video_path, frame_index)

        # resize with padding to given width and height
        resized_image = pil.new(color_image.mode, (self.width, self.height), (0, 0, 0))

        # resize while keeping the aspect ratio
        color_image.thumbnail((self.width, self.height), pil.BILINEAR)
        resized_image.paste(color_image, (
            ((resized_image.width - color_image.width) // 2), (resized_image.height - color_image.height) // 2))
        color_image = resized_image

        if do_flip:
            color_image = color_image.transpose(pil.FLIP_LEFT_RIGHT)

        return color_image

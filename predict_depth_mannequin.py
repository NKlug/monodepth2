import os
import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader

import datasets
import networks
from layers import disp_to_depth
from options import MonodepthOptions
from utils import readlines

splits_dir = os.path.join(os.path.dirname(__file__), "splits")


def predict_depth_mannequin(opt):
    """
    Predict depth maps for given input images.
    @param opt: Options dict
    """

    opt.load_weights_folder = os.path.expanduser(opt.load_weights_folder)

    assert os.path.isdir(opt.load_weights_folder), \
        "Cannot find a folder at {}".format(opt.load_weights_folder)

    print("-> Loading weights from {}".format(opt.load_weights_folder))

    encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
    decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")

    encoder_dict = torch.load(encoder_path)

    filenames = readlines(os.path.join(splits_dir, opt.split, "test_files.txt"))
    # noinspection DuplicatedCode
    data = datasets.SingleVideoMannequinDataset(opt.data_path, filenames,
                                                height=encoder_dict['height'], width=encoder_dict['width'],
                                                frame_idxs=[0], num_scales=4, is_train=False)
    dataloader = DataLoader(data, 16, shuffle=False, num_workers=opt.num_workers,
                            pin_memory=True, drop_last=False)

    encoder = networks.ResnetEncoder(opt.num_layers, False)
    depth_decoder = networks.DepthDecoder(encoder.num_ch_enc)

    model_dict = encoder.state_dict()
    encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
    depth_decoder.load_state_dict(torch.load(decoder_path))

    encoder.cuda()
    encoder.eval()
    depth_decoder.cuda()
    depth_decoder.eval()

    pred_disps = []
    pred_depths = []
    inv_Ks = []
    color_images = []

    width, height = encoder_dict['width'], encoder_dict['height']
    print("-> Computing predictions with size {}x{}".format(
        width, height))

    with torch.no_grad():
        for data in dataloader:
            input_color = data[("color", 0, 0)].cuda()

            if opt.post_process:
                # Post-processed results require each image to have two forward passes
                input_color = torch.cat((input_color, torch.flip(input_color, [3])), 0)

            output = depth_decoder(encoder(input_color))
            pred_disp, pred_depth = disp_to_depth(output[("disp", 0)], opt.min_depth, opt.max_depth)
            pred_disp = pred_disp.cpu()[:, 0].numpy()
            pred_depth = pred_depth.cpu()[:, 0].numpy()
            inv_K = data[("inv_K", 0)].cpu().numpy()

            pred_disps.append(pred_disp)
            pred_depths.append(pred_depth)
            inv_Ks.append(inv_K)
            color_images.append(input_color.cpu().numpy())

    pred_disps = np.concatenate(pred_disps)
    pred_depths = np.concatenate(pred_depths)
    inv_Ks = np.concatenate(inv_Ks)
    color_images = np.concatenate(color_images)

    # clip images and depths to get rid of black bars if they do not have the correct aspect ratio
    # correct aspect ratio is 16/9
    if not np.isclose(width / height, 16 / 9):
        if width > 16 / 9 * height:  # image too wide
            correct_width = round(16 / 9 * height)
            start = (width - correct_width) // 2
            stop = width - start
            pred_disps = pred_disps[:, :, start:stop, ...]
            pred_depths = pred_depths[:, :, start:stop, ...]
            color_images = color_images[:, :, :, start:stop, ...]

        else:  # image too high
            correct_height = round(9 / 16 * width)
            start = (height - correct_height) // 2
            stop = height - start
            pred_disps = pred_disps[:, start:stop, ...]
            pred_depths = pred_depths[:, start:stop, ...]
            color_images = color_images[:, :, start:stop, ...]

    outputs = {
        "depth": pred_depths,
        "disp": pred_disps,
        "inv_K": inv_Ks,
        "color": color_images,
    }

    if opt.save_pred_disps:
        output_path = os.path.join(
            opt.load_weights_folder, "predicted_depths_{}_split.pkl".format(opt.split))
        print("-> Saving predicted disparities to ", output_path)
        with open(output_path, 'wb') as out_file:
            pickle.dump(outputs, out_file)

    return outputs


if __name__ == '__main__':
    options = MonodepthOptions()
    options = options.parse()

    pred_depths = predict_depth_mannequin(options)
    pass

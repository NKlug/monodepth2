import torch
import os
import numpy as np
import cv2

from torch.utils.data import DataLoader

import datasets
import networks
from evaluate_depth import batch_post_process_disparity
from export_gt_depth import export_gt_depths_kitti
from layers import disp_to_depth
from options import MonodepthOptions
from utils import readlines
import pickle

splits_dir = os.path.join(os.path.dirname(__file__), "splits")


def predict_depth(opt):
    """
    Predict depth maps for given input images.
    :param data:
    :param opt:
    """

    opt.load_weights_folder = os.path.expanduser(opt.load_weights_folder)

    assert os.path.isdir(opt.load_weights_folder), \
        "Cannot find a folder at {}".format(opt.load_weights_folder)

    print("-> Loading weights from {}".format(opt.load_weights_folder))

    encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
    decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")

    encoder_dict = torch.load(encoder_path)

    filenames = readlines(os.path.join(splits_dir, options.eval_split, "test_files.txt"))
    # noinspection DuplicatedCode
    data = datasets.KITTIRAWDataset(options.data_path, filenames,
                                    encoder_dict['height'], encoder_dict['width'],
                                    [0], 4, is_train=False)
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
    oxts_list = []
    color_images = []

    print("-> Computing predictions with size {}x{}".format(
        encoder_dict['width'], encoder_dict['height']))

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
            oxts = data[("oxts", 0)]

            if opt.post_process:
                N = pred_disp.shape[0] // 2
                pred_disp = batch_post_process_disparity(pred_disp[:N], pred_disp[N:, :, ::-1])

            pred_disps.append(pred_disp)
            pred_depths.append(pred_depth)
            inv_Ks.append(inv_K)
            oxts_list.append(oxts)
            color_images.append(input_color.cpu().numpy())

    pred_disps = np.concatenate(pred_disps)
    pred_depths = np.concatenate(pred_depths)
    inv_Ks = np.concatenate(inv_Ks)
    color_images = np.concatenate(color_images)

    # get ground truth depth medians for scaling
    split_folder = os.path.join(os.path.dirname(__file__), "splits", opt.split)
    gt_depth_file = os.path.join(split_folder, "gt_depths.npz")

    if not os.path.exists(gt_depth_file):
        print(f"-> No ground truth depths file found. Exporting ground truth depths to {gt_depth_file}!")
        export_gt_depths_kitti(opt)

    print(f"-> Computing per image ground truth depth medians")
    gt_depths = np.load(gt_depth_file, fix_imports=True, encoding='latin1', allow_pickle=True)["data"]
    gt_medians = []
    pred_medians = []
    for i in range(len(gt_depths)):
        gt_depth = gt_depths[i]
        gt_height, gt_width = gt_depth.shape[:2]

        mask = gt_depth > 0
        gt_depth = gt_depth[mask]
        gt_medians.append(np.median(gt_depth))

        pred_depth = pred_depths[i]
        pred_depth = cv2.resize(pred_depth, (gt_width, gt_height))
        pred_depth = pred_depth[mask]

        pred_median = np.median(pred_depth)
        pred_medians.append(pred_median)

    gt_medians = np.asarray(gt_medians)
    pred_medians = np.asarray(pred_medians)

    oxts_list = {key: np.concatenate([value[key] for value in oxts_list]) for key in oxts_list[0]}

    outputs = {
        "depth": pred_depths,
        "disp": pred_disps,
        "inv_K": inv_Ks,
        "oxts": oxts_list,
        "color": color_images,
        "gt_medians": gt_medians,
        "pred_medians": pred_medians
    }

    if opt.save_pred_disps:
        output_path = os.path.join(
            opt.load_weights_folder, "predicted_depths_{}_split.pkl".format(opt.eval_split))
        print("-> Saving predicted disparities to ", output_path)
        with open(output_path, 'wb') as out_file:
            pickle.dump(outputs, out_file)

    return outputs


if __name__ == '__main__':
    options = MonodepthOptions()
    options = options.parse()

    pred_depths = predict_depth(options)
    pass

import torch
import os
import numpy as np

from torch.utils.data import DataLoader

import datasets
import networks
from evaluate_depth import batch_post_process_disparity
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
    inv_Ks = []

    print("-> Computing predictions with size {}x{}".format(
        encoder_dict['width'], encoder_dict['height']))

    with torch.no_grad():
        for data in dataloader:
            input_color = data[("color", 0, 0)].cuda()

            if opt.post_process:
                # Post-processed results require each image to have two forward passes
                input_color = torch.cat((input_color, torch.flip(input_color, [3])), 0)

            output = depth_decoder(encoder(input_color))

            pred_disp, _ = disp_to_depth(output[("disp", 0)], opt.min_depth, opt.max_depth)
            pred_disp = pred_disp.cpu()[:, 0].numpy()
            inv_K = data[("inv_K", 0)].cpu().numpy()

            if opt.post_process:
                N = pred_disp.shape[0] // 2
                pred_disp = batch_post_process_disparity(pred_disp[:N], pred_disp[N:, :, ::-1])

            pred_disps.append(pred_disp)
            inv_Ks.append(inv_K)

    pred_disps = np.concatenate(pred_disps)
    inv_Ks = np.concatenate(inv_Ks)

    outputs = {"depth": pred_disps, "inv_K": inv_Ks}

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

    # import matplotlib as mpl
    # import matplotlib.cm as cm
    # import PIL.Image as pil
    #
    # pred_depth = pred_depths[0]
    # disp_resized_np = pred_depth
    # vmax = np.percentile(disp_resized_np, 95)
    # normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
    # mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
    # colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
    # im = pil.fromarray(colormapped_im)
    # name_dest_im = os.path.join('splits', options.eval_split, "{}_disp.jpeg".format('sequence_test'))
    # im.save(name_dest_im)

    pass

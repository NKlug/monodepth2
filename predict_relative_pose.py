from options import MonodepthOptions
from trainer import Trainer
import numpy as np
import os
import pickle


def predict_relative_pose(opt):
    """
    Estimate relative camera pose between images using the pose network with learned weights.
    """

    trainer = Trainer(opt)
    outputs = []

    print("-> Computing relative pose predictions")

    for batch_idx, inputs in enumerate(trainer.train_loader):
        features = trainer.models["encoder"](inputs["color_aug", 0, 0])

        for key, ipt in inputs.items():
            inputs[key] = ipt.to(trainer.device)

        outputs.append(trainer.predict_poses(inputs, features))

    output_list = {key: np.concatenate([value[key] for value in outputs]) for key in outputs[0]}

    if opt.save_pred_disps:
        output_path = os.path.join(
            opt.load_weights_folder, "predicted_relative_poses_{}_split.pkl".format(opt.eval_split))
        print("-> Saving predicted disparities to ", output_path)
        with open(output_path, 'wb') as out_file:
            pickle.dump(output_list, out_file)

    return output_list


if __name__ == '__main__':
    options = MonodepthOptions()
    options = options.parse()

    pred_depths = predict_relative_pose(options)
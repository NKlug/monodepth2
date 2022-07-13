import os
import glob

if __name__ == '__main__':
    with open('depth_prediction_sequences.sh', 'w') as file:
        for split in glob.glob('../splits/*_drive_*'):
            split_name = os.path.basename(split)
            command = f'python ../predict_depth.py ' \
                      f'--load_weights_folder /data/training/monodepth2/mono_model/models/weights_19/ ' \
                      f'--data_path /datasets/kitti_data_jpg/ --split {split_name} --save_pred_disps\n\n'
            file.write(command)

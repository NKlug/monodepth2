# How to visualize

0. Train network or use pretrained weights.
1. Create desired data split (i.e. a sequence from a single drive) or use an existing split from the `splits` directory
(e.g. '2011_09_26_drive_0001').
2. Run `predict_depth.py` locally or remotly with the desired split, i.e. 
 ```
python3 predict_depth.py
--load_weights_folder
<model_directory>/mono_model/models/weights_19/
--data_path
/datasets/kitti_data_jpg/
--split
2011_09_26_drive_0001
--save_pred_disps
```
3. If run remotely, download the generated depth predictions located at `<model_dir>/models/<model name>/<weight folder>/predicted_depths_<split name>_split.pkl` to `models/<model name>/<weight folder>`.
4. Run `visualize.py` with respective options, i.e.
```
python3 visualize.py
--load_weights_folder 
models/mono_model/models/weights_19
 --split sequence
```
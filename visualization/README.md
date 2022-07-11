# How to visualize

1. Create desired data split (i.e. a sequence from a single drive) or use an existing split.
2. Run `predict_depth.py` remotely with desired split.
3. Download the generated depth predictions located at `<model_dir>/models/<model name>/<weight folder>/predixted_depths_<split name>_split.pkl` to `models/<model name>/<weight folder>`.
4. Run `visualize.py` with respective options, i.e. split name etc.
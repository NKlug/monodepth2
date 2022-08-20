We subsample the provided frames to use at most 5 frames per second.
This ensures that there is enough change in perspective for the network to 
learn properly.
Hence, for training, make sure to use frame_idxs=[0, 6, -6] (30fps / 5 fps)
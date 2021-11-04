# use SGDepth code for depth expert - https://github.com/xavysp/DexiNed/blob/master/DexiNed-Pytorch/
import os

import numpy as np
import torch
import torch.nn.functional as F

from experts.basic_expert import BasicExpert

W, H = 256, 256

current_dir_name = os.path.dirname(os.path.realpath(__file__))
edges_model_path = os.path.join(current_dir_name, 'models/edges_dexined.h5')


class EdgesModel(BasicExpert):
    def __init__(self, full_expert=True):
        if full_expert:
            from experts.edges.model import DexiNed
            checkpoint_path = edges_model_path
            device = "gpu" if torch.cuda.is_available() else "cpu"
            self.device = device
            rgbn_mean = np.array([103.939, 116.779, 123.68,
                                  137.86])[None, None, None, :]
            input_shape = (1, H, W, 3)
            self.model = DexiNed(rgb_mean=rgbn_mean)
            self.model.build(input_shape=input_shape)
            self.model.load_weights(checkpoint_path)

        self.domain_name = "edges"
        self.n_maps = 1
        self.str_id = "dexined"
        self.identifier = self.domain_name + "_" + self.str_id

    def apply_expert_batch(self, batch_rgb_frames):
        edge_maps = []
        batch_rgb_frames = batch_rgb_frames.numpy().astype(np.float32)
        preds = self.model(batch_rgb_frames, training=False)
        edge_maps = torch.sigmoid(torch.from_numpy(
            preds.numpy())).numpy()[:, :, :, 0]
        edge_maps = edge_maps[:, None, :, :]
        edge_maps = edge_maps.astype('float32')
        return edge_maps

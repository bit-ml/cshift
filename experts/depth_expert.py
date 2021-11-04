# use SGDepth code for depth expert - https://github.com/ifnspaml/SGDepth
import os

import numpy as np
import torch

from experts.basic_expert import BasicExpert
from experts.xtc.unet import UNet

W, H = 256, 256

current_dir_name = os.path.dirname(os.path.realpath(__file__))
depthxtc_model_path = os.path.join(current_dir_name, 'models/depth_xtc.pth')


class DepthModelXTC(BasicExpert):
    def __init__(self, full_expert=True):
        if full_expert:
            model_path = depthxtc_model_path
            self.model = UNet(downsample=6, out_channels=1)
            model_state_dict = torch.load(model_path)
            self.model.load_state_dict(model_state_dict)
            self.model.eval()

            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.device = device
            self.model.to(device)

        self.domain_name = "depth"
        self.n_maps = 1
        self.str_id = "xtc"
        self.identifier = self.domain_name + "_" + self.str_id

    def apply_expert_batch(self, batch_rgb_frames):
        batch_rgb_frames = batch_rgb_frames.permute(0, 3, 1, 2) / 255.
        depth_maps, _ = self.model(batch_rgb_frames.to(self.device))
        depth_maps = depth_maps.data.cpu().numpy()
        depth_maps = np.array(depth_maps).astype('float32')
        return depth_maps

    def test_gt(self, loss_fct, pred, target):
        l_target = target.clone()

        is_nan = target != target
        bm = ~is_nan

        l_target[is_nan] = 0
        l_target = l_target * bm
        l_pred = pred * bm

        loss = loss_fct(l_pred, l_target)
        return loss

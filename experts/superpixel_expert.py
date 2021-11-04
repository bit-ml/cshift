import os

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from experts.basic_expert import BasicExpert
from experts.spixel.train_util import shift9pos

current_dir_name = os.path.dirname(os.path.realpath(__file__))
sp_model_path = os.path.join(current_dir_name, 'models/SpixelNet_bsd_ckpt.tar')

H, W = 256, 256
DOWNSIZE = 16


class SuperPixel(BasicExpert):
    def __init__(self, full_expert=True):
        if full_expert:
            from experts.spixel.Spixel_single_layer import SpixelNet1l_bn

            # create model
            network_data = torch.load(sp_model_path)
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

            self.model = SpixelNet1l_bn(data=network_data).to(self.device)
            self.model.eval()
            cudnn.benchmark = True
            self.mean_values = torch.tensor([0.411, 0.432,
                                             0.45]).view(1, 3, 1,
                                                         1).to(self.device)
            self.grayw = torch.tensor([0.2989, 0.5870,
                                       0.1140]).to(self.device)[None, :, None,
                                                                None]

        self.domain_name = "superpixel"
        self.n_maps = 1

        self.str_id = "fcn"
        self.identifier = self.domain_name + "_" + self.str_id

    def apply_expert_batch(self, batch_rgb_frames):
        batch_rgb_frames = batch_rgb_frames.to(self.device)
        batch_rgb_frames = batch_rgb_frames.permute(0, 3, 1, 2) / 255.

        # get spixel id
        n_spixl_h, n_spixl_w = int(H / DOWNSIZE), int(W / DOWNSIZE)

        spix_values = np.int32(
            np.arange(0, n_spixl_w * n_spixl_h).reshape(
                (n_spixl_h, n_spixl_w)))
        spix_idx_tensor_ = shift9pos(spix_values)

        spix_idx_tensor = np.repeat(np.repeat(spix_idx_tensor_,
                                              DOWNSIZE,
                                              axis=1),
                                    DOWNSIZE,
                                    axis=2)

        spixeIds = torch.from_numpy(np.tile(
            spix_idx_tensor, (1, 1, 1, 1))).type(torch.float).cuda()

        # NN forward
        preproc_batch = batch_rgb_frames - self.mean_values
        spixel_outp = self.model(preproc_batch)

        # assign the spixel map
        curr_spixl_map = update_spixl_map(spixeIds, spixel_outp)
        ori_sz_spixel_map = F.interpolate(curr_spixl_map.type(torch.float),
                                          size=(H, W),
                                          mode='nearest').type(torch.int)

        # n_spixel = int(n_spixl_h * n_spixl_w)
        # spixel_viz, spixel_label_map_old = get_spixel_image(
        #     batch_rgb_frames[0].clamp(0, 1),
        #     ori_sz_spixel_map[0].squeeze(),
        #     n_spixels=n_spixel,
        #     b_enforce_connect=True)

        gray_batch_rgb_frames = (batch_rgb_frames * self.grayw).sum(axis=1)
        bs = batch_rgb_frames.shape[0]
        spixel_label_map = postproc_labels(ori_sz_spixel_map[:, 0])
        for label in np.unique(spixel_label_map):
            # mask = (spixel_label_map[:, None].repeat(axis=1, repeats=3) == label)
            mask = torch.from_numpy(spixel_label_map == label).to(self.device)
            for idx_batch in range(bs):
                gray_batch_rgb_frames[idx_batch][
                    mask[idx_batch]] = gray_batch_rgb_frames[idx_batch][
                        mask[idx_batch]].mean()

        spixel_maps = np.array(gray_batch_rgb_frames[:,
                                                     None].data.cpu().numpy())
        return spixel_maps

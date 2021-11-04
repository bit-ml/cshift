# https://github.com/philgyford/python-halftone
import numpy as np
import torch
from PIL import Image

from experts.basic_expert import BasicExpert
from experts.halftone.halftone import Halftone

W, H = 256, 256


class HalftoneModel(BasicExpert):
    def __init__(self, full_expert=True, style=0):
        # if full_expert:
        # self.model = Halftone()
        self.style = style
        self.classes_weights = None

        if style == 0:
            self.n_maps = 1
            self.domain_name = "halftone_gray"
        elif style == 1:
            self.n_maps = 3
            self.domain_name = "halftone_rgb"
        elif style == 2:
            self.n_maps = 4
            self.domain_name = "halftone_cmyk"
        elif style == 3:
            self.n_maps = 1
            self.domain_name = "halftone_rot_gray"
        self.str_id = "basic"
        self.identifier = self.domain_name  # + "_" + self.str_id

    def apply_expert_batch(self, batch_rgb_frames):
        halftone_maps = []
        for idx, rgb_frame in enumerate(batch_rgb_frames):
            rgb_frame = Image.fromarray(np.uint8(rgb_frame))
            halftone_map = self.apply_expert_one_frame(rgb_frame)
            halftone_maps.append(np.array(halftone_map))
        halftone_maps = np.array(halftone_maps).astype('float32')
        return halftone_maps

    def apply_expert_one_frame(self, rgb_frame):
        resized_rgb_frame = rgb_frame.resize((W, H))
        halftone_map = Halftone(resized_rgb_frame, self.style).make()

        if self.style == 0 or self.style == 3:
            halftone_map = np.array(halftone_map)[None, :, :] / 255.
        else:
            halftone_map = np.array(halftone_map).transpose(2, 0, 1) / 255.

        return halftone_map

    def get_task_type(self):
        return BasicExpert.TASK_CLASSIFICATION

    def no_maps_as_nn_input(self):
        return 1

    def no_maps_as_nn_output(self):
        return 2

    def no_maps_as_ens_input(self):
        return 2

    def gt_train_transform(self, gt_maps):
        return gt_maps.squeeze(1).long()

    def gt_eval_transform(self, gt_maps, n_classes):
        return gt_maps.squeeze(1).long()

    def exp_eval_transform(self, gt_maps, n_classes):
        return gt_maps.squeeze(1).long()

    def gt_to_inp_transform(self, inp_1chan_cls, n_classes):
        inp_1chan_cls = inp_1chan_cls.squeeze(1).long()
        bs, h, w = inp_1chan_cls.shape

        outp_multichan = torch.zeros(
            (bs, n_classes, h, w)).to(inp_1chan_cls.device).float()
        for chan in range(n_classes):
            outp_multichan[:, chan][inp_1chan_cls == chan] = 1.
        return outp_multichan

    def postprocess_eval(self, nn_outp):
        return nn_outp

    def postprocess_ensemble_eval(self, nn_outp):
        return nn_outp

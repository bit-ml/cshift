import torch
from skimage.color import rgb2hsv

from experts.basic_expert import BasicExpert


class HSVExpert(BasicExpert):
    def __init__(self, full_expert=True):
        self.n_maps = 3
        self.domain_name = "hsv"
        self.str_id = ""
        self.identifier = self.domain_name

    def apply_expert_batch(self, batch_rgb_frames):
        batch_rgb_frames = batch_rgb_frames
        outp_maps = rgb2hsv(batch_rgb_frames)
        hsv_maps = outp_maps.astype('float32').transpose(0, 3, 1, 2) / 255.
        return hsv_maps

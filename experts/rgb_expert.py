import numpy as np

from experts.basic_expert import BasicExpert

W, H = 256, 256


class RGBModel(BasicExpert):
    def __init__(self, full_expert=True):
        self.domain_name = "rgb"
        self.n_maps = 3
        self.str_id = ""
        self.identifier = "rgb"

    def apply_expert_batch(self, batch_rgb_frames):
        batch_rgb_frames = batch_rgb_frames.numpy()
        batch_rgb_frames = batch_rgb_frames.astype('float32')
        batch_rgb_frames = batch_rgb_frames / 255.0
        batch_rgb_frames = np.moveaxis(batch_rgb_frames, 3, 1)
        return batch_rgb_frames

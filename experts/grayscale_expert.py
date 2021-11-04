import torch

from experts.basic_expert import BasicExpert


class Grayscale(BasicExpert):
    def __init__(self, full_expert=True):
        self.n_maps = 1
        self.domain_name = "grayscale"
        self.str_id = ""
        self.identifier = self.domain_name
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.w = torch.tensor([0.2989, 0.5870, 0.1140]).to(device)[None, :,
                                                                   None, None]

    def apply_expert_batch(self, batch_rgb_frames):
        batch_rgb_frames = batch_rgb_frames.permute(0, 3, 1, 2) / 255.
        inp = batch_rgb_frames.to(self.device)
        grayscale_maps = (inp * self.w).sum(axis=1, keepdim=True)
        grayscale_maps = grayscale_maps.data.cpu().numpy().astype('float32')
        return grayscale_maps

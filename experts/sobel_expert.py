import torch
from utils.utils import EPSILON, get_gaussian_filter

from experts.basic_expert import BasicExpert


class SobelEdgesExpert(BasicExpert):
    def __init__(self, sigma, full_expert=True):
        self.domain_name = "edges"
        self.str_id = "sobel"
        self.identifier = self.domain_name + "_" + self.str_id
        self.n_maps = 1
        self.sigma = sigma
        self.win_size = max(2 * (int(2.0 * sigma + 0.5)) + 1, 3)
        self.n_channels = 1

        if full_expert:
            with torch.no_grad():
                # GAUSSIAN
                self.g_filter_value = get_gaussian_filter(
                    n_channels=self.n_channels,
                    win_size=self.win_size,
                    sigma=sigma).float()
                self.g_filter = torch.nn.Conv2d(kernel_size=(self.win_size,
                                                             self.win_size),
                                                in_channels=1,
                                                out_channels=1,
                                                padding=self.win_size // 2,
                                                padding_mode='reflect')
                self.g_filter.weight.requires_grad = False
                self.g_filter.weight.copy_(self.g_filter_value)
                self.g_filter.bias.requires_grad = False
                self.g_filter.bias.zero_()

                # SOBEL
                self.sobel_filter_value = torch.FloatTensor([[1, 2, 1],
                                                             [0, 0, 0],
                                                             [-1, -2,
                                                              -1]])[None, None]
                # Sobel_X
                self.sobel_filter_x = torch.nn.Conv2d(kernel_size=(3, 3),
                                                      in_channels=1,
                                                      out_channels=1,
                                                      padding=1,
                                                      padding_mode='reflect')
                self.sobel_filter_x.weight.requires_grad = False
                self.sobel_filter_x.weight.copy_(self.sobel_filter_value)
                self.sobel_filter_x.bias.requires_grad = False
                self.sobel_filter_x.bias.zero_()

                # Sobel_Y
                self.sobel_filter_y = torch.nn.Conv2d(kernel_size=(3, 3),
                                                      in_channels=1,
                                                      out_channels=1,
                                                      padding=1,
                                                      padding_mode='reflect')
                self.sobel_filter_y.weight.requires_grad = False
                self.sobel_filter_y.weight.copy_(
                    self.sobel_filter_value.permute((0, 1, 3, 2)))
                self.sobel_filter_y.bias.requires_grad = False
                self.sobel_filter_y.bias.zero_()

    def apply_expert_batch(self, batch_rgb_frames):
        batch_rgb_frames = batch_rgb_frames.permute(0, 3, 1, 2) / 255.
        batch_rgb_frames = batch_rgb_frames.mean(axis=1, keepdim=True)

        blurred = self.g_filter(batch_rgb_frames)
        sx = self.sobel_filter_x(blurred)
        sy = self.sobel_filter_y(blurred)

        edges = torch.hypot(sx, sy)
        edges.clamp_(max=1)
        edges /= (edges.amax(dim=(2, 3), keepdim=True) + EPSILON)

        return edges.data.cpu().numpy()


class SobelEdgesExpertSigmaLarge(SobelEdgesExpert):
    def __init__(self, full_expert=True):
        SobelEdgesExpert.__init__(self, sigma=4., full_expert=full_expert)
        self.str_id = "sobel_large"
        self.identifier = self.str_id


class SobelEdgesExpertSigmaMedium(SobelEdgesExpert):
    def __init__(self, full_expert=True):
        SobelEdgesExpert.__init__(self, sigma=1., full_expert=full_expert)
        self.str_id = "sobel_medium"
        self.identifier = self.str_id


class SobelEdgesExpertSigmaSmall(SobelEdgesExpert):
    def __init__(self, full_expert=True):
        SobelEdgesExpert.__init__(self, sigma=0.1, full_expert=full_expert)
        self.str_id = "sobel_small"
        self.identifier = self.str_id

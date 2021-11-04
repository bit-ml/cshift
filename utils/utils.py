import os
import sys
from math import exp

import lpips
import numpy as np
import torch
import torch.nn.functional as F
from skimage import color
from torch import nn

sys.path.insert(0,
                os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

EPSILON = 0.00001
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
COLORS_SHORT = ('red', 'blue', 'yellow', 'magenta', 'green', 'indigo',
                'darkorange', 'cyan', 'pink', 'yellowgreen', 'chocolate',
                'lightsalmon', 'lime', 'silver', 'gainsboro', 'gold', 'coral',
                'aquamarine', 'lightcyan', 'oldlace', 'darkred', 'snow')

BIG_VALUE = 1000


def img_for_plot(img, dst_id):
    '''
    img shape NCHW, ex: torch.Size([3, 1, 256, 256])
    '''
    img = img.clone()
    n, c, _, _ = img.shape
    if dst_id.find("halftone") >= 0:
        if img.shape[1] > 1:
            tasko_labels = img.argmax(dim=1, keepdim=True)
        else:
            tasko_labels = img

        img = tasko_labels
        c = 1
    elif dst_id.find("sem_seg") >= 0:

        if img.shape[1] > 1:
            tasko_labels = img.argmax(dim=1, keepdim=True)
        else:
            tasko_labels = img
        all_classes = 12
        for idx in range(all_classes):
            tasko_labels[:, 0, 0, idx] = idx
            tasko_labels[:, 0, idx, 0] = idx

        result = color.label2rgb((tasko_labels[:, 0]).data.cpu().numpy(),
                                 colors=COLORS_SHORT,
                                 bg_label=0).transpose(0, 3, 1, 2)
        img = torch.from_numpy(result.astype(np.float32)).contiguous()
        c = 3
    return img


def get_gaussian_filter(n_channels, win_size, sigma):
    # build gaussian filter for SSIM
    h_win_size = win_size // 2
    yy, xx = torch.meshgrid([
        torch.arange(-h_win_size, h_win_size + 1, dtype=torch.float32),
        torch.arange(-h_win_size, h_win_size + 1, dtype=torch.float32)
    ])
    g_filter = torch.exp((-0.5) * ((xx**2 + yy**2) / (2 * sigma**2)))
    g_filter = g_filter.unsqueeze(0).unsqueeze(0)
    g_filter = g_filter.repeat(n_channels, 1, 1, 1)
    g_filter = g_filter / torch.sum(g_filter)
    return g_filter


class SSIMLoss(torch.nn.Module):
    def __init__(self, n_channels, win_size, reduction='mean'):
        super(SSIMLoss, self).__init__()
        self.n_channels = n_channels
        self.win_size = win_size
        self.sigma = self.win_size / 7
        self.reduction = reduction
        self.g_filter = get_gaussian_filter(self.n_channels, self.win_size,
                                            self.sigma).to(device)

    def forward(self, batch1, batch2):

        mu1 = torch.nn.functional.conv2d(batch1,
                                         self.g_filter,
                                         padding=self.win_size // 2,
                                         groups=self.n_channels)
        mu2 = torch.nn.functional.conv2d(batch2,
                                         self.g_filter,
                                         padding=self.win_size // 2,
                                         groups=self.n_channels)
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = torch.nn.functional.conv2d(batch1 * batch1,
                                               self.g_filter,
                                               padding=self.win_size // 2,
                                               groups=self.n_channels) - mu1_sq
        sigma1_sq = torch.abs(sigma1_sq)

        sigma2_sq = torch.nn.functional.conv2d(batch2 * batch2,
                                               self.g_filter,
                                               padding=self.win_size // 2,
                                               groups=self.n_channels) - mu2_sq
        sigma2_sq = torch.abs(sigma2_sq)

        sigma12 = torch.nn.functional.conv2d(batch1 * batch2,
                                             self.g_filter,
                                             padding=self.win_size // 2,
                                             groups=self.n_channels) - mu1_mu2

        C1 = 0.01**2
        C2 = 0.03**2

        ssim_map = ((2 * mu1_mu2 + C1) *
                    (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                           (sigma1_sq + sigma2_sq + C2))
        if self.reduction == 'mean':
            res = ssim_map.view((ssim_map.shape[0], ssim_map.shape[1],
                                 -1)).mean(2).mean(1).mean()
        res = 1 - ((res + 1) / 2)
        return res


class DummySummaryWriter:
    def __init__(*args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return self

    def __getattr__(self, *args, **kwargs):
        return self


class VarianceScore(nn.Module):
    def __init__(self, reduction=False):
        super(VarianceScore, self).__init__()

    def forward(self, batch):
        avg_b = torch.mean(batch, 0)[None]
        avg_b = (batch - avg_b)**2
        avg_b = torch.mean(avg_b, 0)
        return avg_b


class WeightedVarianceScore(nn.Module):
    def __init__(self, reduction=False):
        super(WeightedVarianceScore, self).__init__()

    def forward(self, batch, weights):
        # batch, weights - bs x n_chs x h x w x n_exps
        avg = torch.mean(batch * weights, dim=4, keepdim=True)
        variance = torch.sum(weights * (batch - avg)**2, dim=4, keepdim=True)
        s = torch.sum(weights, dim=4, keepdim=True)
        s[s == 0] = 1
        variance = variance / s
        return variance


class MeanScoreFunction(nn.Module):
    def __init__(self):
        super(MeanScoreFunction, self).__init__()

    def compute_distances(self, data):
        mean = data.mean(dim=-1, keepdim=True)
        distance_maps = torch.abs(data - mean)

        return distance_maps

    def update_distances(self, data, weights):
        sum_weights = weights.sum(axis=-1)
        sum_weights[sum_weights == 0] = 1

        w_mean = ((data * weights).sum(axis=-1) / sum_weights)[..., None]
        distance_maps = torch.abs(data - w_mean)

        return distance_maps


class ScoreFunctions(nn.Module):
    def compute_distances(self, data):
        '''
            twd_expert_distances
        '''
        bs, n_chs, h, w, n_tasks = data.shape
        distance_maps = []
        for i in range(n_tasks - 1):
            distance_map = self.forward(data[..., -1], data[..., i])
            distance_maps.append(distance_map)

        # add expert vs expert
        distance_maps.append(torch.zeros_like(distance_map))

        distance_maps = torch.stack(distance_maps, 0).permute(1, 2, 3, 4, 0)
        return distance_maps

    def forward(self, batch1, batch2):
        pass


class SimScore_SSIM(ScoreFunctions):
    def __init__(self, n_channels, win_size, reduction=False):
        super(SimScore_SSIM, self).__init__()
        self.n_channels = n_channels
        self.win_size = win_size
        self.sigma = self.win_size / 7
        self.reduction = reduction
        self.g_filter = nn.Parameter(get_gaussian_filter(
            self.n_channels, self.win_size, self.sigma),
                                     requires_grad=False)
        self.conv_filter = torch.nn.Conv2d(in_channels=n_channels,
                                           out_channels=n_channels,
                                           kernel_size=win_size,
                                           padding=win_size // 2,
                                           padding_mode='replicate',
                                           groups=n_channels,
                                           bias=False)
        self.conv_filter.weight = self.g_filter

    def forward(self, batch1, batch2):
        mu1 = self.conv_filter(batch1)
        mu2 = self.conv_filter(batch2)
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        sigma1_sq = self.conv_filter(batch1 * batch1) - mu1_sq
        sigma1_sq = torch.abs(sigma1_sq)
        sigma2_sq = self.conv_filter(batch2 * batch2) - mu2_sq
        sigma2_sq = torch.abs(sigma2_sq)
        sigma12 = self.conv_filter(batch1 * batch2) - mu1_mu2
        C1 = 0.01**2
        C2 = 0.03**2
        ssim_map = ((2 * mu1_mu2 + C1) *
                    (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                           (sigma1_sq + sigma2_sq + C2))
        if self.reduction:
            sim_score = ssim_map.view((ssim_map.shape[0], ssim_map.shape[1],
                                       -1)).mean(2).mean(1).sum()
        else:
            sim_score = ssim_map

        # there seem to be small numerical issues
        sim_score = torch.clamp(sim_score, -1, 1)

        sim_score = (sim_score + 1) / 2
        return 1 - sim_score


class SimScore_L1(ScoreFunctions):
    def __init__(self, reduction=False):
        super(SimScore_L1, self).__init__()
        if reduction:
            self.l1 = torch.nn.L1Loss(reduction='mean')
        else:
            self.l1 = torch.nn.L1Loss(reduction='none')

    def forward(self, batch1, batch2):
        res = self.l1(batch1, batch2)
        # => res >= 0, with 0 best
        return res


class SimScore_L2(ScoreFunctions):
    def __init__(self, reduction=False):
        super(SimScore_L2, self).__init__()
        if reduction:
            self.l2 = torch.nn.MSELoss(reduction='mean')
        else:
            self.l2 = torch.nn.MSELoss(reduction='none')

    def forward(self, batch1, batch2):
        res = self.l2(batch1, batch2)
        # => res >= 0, with 0 best
        return res


class SimScore_Equal(ScoreFunctions):
    def __init__(self):
        super(SimScore_Equal, self).__init__()
        self.zeros = torch.zeros((70, 3, 256, 256)).cuda()

    def forward(self, batch1, batch2):
        bs, n_chs, h, w = batch1.shape
        bso, n_chso, ho, wo = self.zeros.shape
        if bso == bs and n_chso == n_chs and ho == h and wo == w:
            return self.zeros

        del self.zeros
        self.zeros = torch.zeros_like(batch1)
        return self.zeros


class SimScore_PSNR(ScoreFunctions):
    def __init__(self, reduction=False):
        super(SimScore_PSNR, self).__init__()
        if reduction:
            self.reduction = 'mean'
        else:
            self.reduction = 'none'

    def forward(self, batch1, batch2):
        mse = torch.nn.functional.mse_loss(batch1,
                                           batch2,
                                           reduction=self.reduction)
        norm_dist = torch.log10(1 / (mse + EPSILON))
        return norm_dist


class SimScore_LPIPS(ScoreFunctions):
    def __init__(self, n_channels):
        super(SimScore_LPIPS, self).__init__()
        self.n_channels = n_channels
        self.lpips_net = lpips.LPIPS(net='squeeze',
                                     spatial=True,
                                     verbose=False)
        self.lpips_net.eval()
        self.lpips_net.requires_grad_(False)

    def forward(self, batch1, batch2):
        n_chn = batch1.shape[1]
        if n_chn in [1, 3]:
            distance = self.lpips_net.forward(batch1, batch2)
            return distance.repeat(1, n_chn, 1, 1)
        else:
            distance = torch.zeros_like(batch1)
            for chan in range(n_chn):
                distance[:, chan:chan + 1] = self.lpips_net.forward(
                    batch1[:, chan:chan + 1], batch2[:, chan:chan + 1])
            return distance


class SimScore_LPIPS_per_channel(ScoreFunctions):
    def __init__(self, n_channels):
        super(SimScore_LPIPS_per_channel, self).__init__()
        self.n_channels = n_channels
        self.lpips_net = lpips.LPIPS(net='squeeze',
                                     spatial=True,
                                     verbose=False)
        self.lpips_net.eval()
        self.lpips_net.requires_grad_(False)

    def forward(self, batch1, batch2):
        n_chn = batch1.shape[1]
        distance = torch.zeros_like(batch1)
        for chan in range(n_chn):
            distance[:, chan:chan + 1] = self.lpips_net.forward(
                batch1[:, chan:chan + 1], batch2[:, chan:chan + 1])
        return distance


class EnsembleFilter_TwdExpert(torch.nn.Module):
    def __init__(self,
                 n_channels,
                 dst_domain_name,
                 postprocess_eval,
                 similarity_fcts=['ssim'],
                 kernel_fct='gauss',
                 comb_type='mean',
                 thresholds=[0.5]):
        super(EnsembleFilter_TwdExpert, self).__init__()
        self.thresholds = thresholds
        self.similarity_fcts = similarity_fcts
        self.n_channels = n_channels
        self.dst_domain_name = dst_domain_name
        self.postprocess_eval = postprocess_eval

        if kernel_fct == 'flat':
            self.kernel = self.kernel_flat
        elif kernel_fct == 'flat_weighted':
            self.kernel = self.kernel_flat_weighted
        elif kernel_fct == 'gauss':
            self.kernel = self.kernel_gauss

        if comb_type == 'mean':
            self.ens_aggregation_fcn = self.forward_mean
        else:
            self.ens_aggregation_fcn = self.forward_median
        sim_models = []
        for sim_fct in similarity_fcts:
            if sim_fct == 'ssim':
                sim_model = SimScore_SSIM(n_channels, 11)
            elif sim_fct == 'l1':
                sim_model = SimScore_L1()
            elif sim_fct == 'l2':
                sim_model = SimScore_L2()
            elif sim_fct == 'equal':
                sim_model = SimScore_Equal()
            elif sim_fct == 'psnr':
                sim_model = SimScore_PSNR()
            elif sim_fct == 'lpips':
                sim_model = SimScore_LPIPS(n_channels)
            elif sim_fct == 'dist_mean':
                sim_model = MeanScoreFunction()
            elif sim_fct == 'lpips_per_channel':
                sim_model = SimScore_LPIPS_per_channel(n_channels)
            else:
                assert (False)

            sim_models.append(sim_model)
        self.distance_models = torch.nn.ModuleList(sim_models)

    def forward_mean(self, data, weights):
        data = data * weights.cuda()
        return torch.sum(data, -1)

    def forward_median(self, data, weights):
        bs, n_chs, h, w, n_exps = data.shape
        fwd_result = torch.zeros_like(data[..., 0])
        for chan in range(n_chs):
            data_chan = data[:, chan].contiguous()
            data_chan = data_chan.view(bs * h * w, n_exps)
            weights_chan = weights[:, chan].contiguous()
            weights_chan = weights_chan.view(bs * h * w, n_exps)
            indices = torch.argsort(data_chan, 1)

            data_chan = data_chan[torch.arange(bs * h * w).unsqueeze(1).repeat(
                (1, n_exps)), indices]
            weights_chan = weights_chan[torch.arange(bs * h *
                                                     w).unsqueeze(1).repeat(
                                                         (1, n_exps)), indices]
            weights_chan = torch.cumsum(weights_chan, 1)

            weights_chan[weights_chan < 0.5] = 2
            _, indices = torch.min(weights_chan, dim=1, keepdim=True)
            data_chan = data_chan[torch.arange(bs * h * w).unsqueeze(1),
                                  indices]
            data_chan = data_chan.view(bs, h, w)
            fwd_result[:, chan] = data_chan

        return fwd_result

    def scale_distance_maps(self, distance_maps):
        bm = distance_maps == BIG_VALUE
        distance_maps[bm] = 0
        max_val = torch.amax(distance_maps, axis=(2, 3, 4), keepdim=True)
        min_val = torch.amin(distance_maps, axis=(2, 3, 4), keepdim=True)
        distance_maps = (distance_maps - min_val) / (max_val - min_val +
                                                     EPSILON)
        distance_maps[bm] = 1
        return distance_maps

    def kernel_flat(self, chan_sim_maps, meanshift_iter):
        # indicates what we want to remove
        chan_mask = chan_sim_maps > self.thresholds[meanshift_iter]
        chan_sim_maps[chan_mask] = 0
        chan_sim_maps[~chan_mask] = 1
        return chan_sim_maps

    def kernel_flat_weighted(self, chan_sim_maps, meanshift_iter):
        # indicates what we want to remove
        chan_mask = chan_sim_maps > self.thresholds[meanshift_iter]
        chan_sim_maps = 1 - chan_sim_maps
        chan_sim_maps[chan_mask] = 0
        return chan_sim_maps

    def kernel_gauss(self, chan_sim_maps, meanshift_iter):
        chan_sim_maps = torch.exp(-((chan_sim_maps**2) /
                                    (2 * self.thresholds[meanshift_iter]**2)))
        return chan_sim_maps

    def forward(self, data):
        for meanshift_iter in range(len(self.thresholds)):
            bs, n_chs, h, w, n_tasks = data.shape
            distance_maps = torch.zeros_like(data)

            # combine multiple similarities functions
            for dist_idx, dist_model in enumerate(self.distance_models):
                distance_map = dist_model.compute_distances(data)
                distance_map = self.scale_distance_maps(distance_map)
                distance_maps += distance_map
            if len(self.distance_models) > 1:
                distance_maps = self.scale_distance_maps(distance_maps)

            # kernel: transform distances to similarities
            for chan in range(n_chs):
                distance_maps[:, chan] = self.kernel(distance_maps[:, chan],
                                                     meanshift_iter)

            # sum = 1, similarity maps in fact
            sum_ = torch.sum(distance_maps, dim=-1, keepdim=True)
            sum_[sum_ == 0] = 1
            distance_maps = distance_maps / sum_

            ensemble_result = self.ens_aggregation_fcn(data, distance_maps)

            # clamp/other the ensemble
            ensemble_result = self.postprocess_eval(ensemble_result)

            data[..., -1] = ensemble_result
        return ensemble_result

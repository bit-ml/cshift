# https://github.com/EPFL-VILAB/XTConsistency
import os

import torch

from experts.basic_expert import BasicExpert
from experts.xtc.unet import UNet

W, H = 256, 256

current_dir_name = os.path.dirname(os.path.realpath(__file__))
normals_model_path = os.path.join(current_dir_name, 'models/normals_xtc.pth')


class SurfaceNormalsXTC(BasicExpert):
    SOME_THRESHOLD = 0.

    def __init__(self, dataset_name, full_expert=True, no_alt=False):
        '''
            dataset_name: "taskonomy" or "replica"
            no_alt: if True -> we keep the last channel 
        '''
        if full_expert:
            #         from tensorflow.python.keras.utils.generic_utils import \
            # _SKIP_FAILED_SERIALIZATION
            model_path = normals_model_path
            self.model = UNet()
            model_state_dict = torch.load(model_path)
            self.model.load_state_dict(model_state_dict)
            self.model.eval()
            self.dataset_name = dataset_name
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.device = device
            self.model.to(device)

        self.domain_name = "normals"
        if no_alt:
            self.domain_name += '_no_alt'

        self.n_final_maps = 3

        self.chan_gen_fcn = torch.ones_like
        self.n_maps = 2

        self.str_id = "xtc"
        self.identifier = self.domain_name + "_" + self.str_id
        self.no_alt = no_alt

    def apply_expert_batch(self, batch_rgb_frames):
        batch_rgb_frames = batch_rgb_frames.permute(0, 3, 1, 2) / 255.
        out_maps = self.model(batch_rgb_frames.to(self.device))[0]

        if not self.no_alt:
            # 1. CLAMP
            torch.clamp_(out_maps[:, :2], min=0, max=1)
            torch.clamp_(out_maps[:, 2], min=0., max=0.5)

            # 2. ALIGN ranges
            out_maps[:, 2] += 0.5

            # 4. NORMALIZE it

            out_maps = out_maps * 2 - 1
            out_maps[:, 2] = SurfaceNormalsXTC.SOME_THRESHOLD
            norm_normals_maps = torch.norm(out_maps, dim=1, keepdim=True)
            norm_normals_maps[norm_normals_maps == 0] = 1
            out_maps = out_maps / norm_normals_maps
            out_maps = (out_maps + 1) / 2
        else:
            torch.clamp_(out_maps, min=-1, max=1)

        out_maps = out_maps.data.cpu().numpy().astype('float32')

        return out_maps

    def no_maps_as_nn_input(self):
        return self.n_final_maps

    def no_maps_as_nn_output(self):
        return self.n_maps

    def no_maps_as_ens_input(self):
        return self.n_final_maps

    def postprocess_eval(self, nn_outp):
        bs, nch, h, w = nn_outp.shape

        if nch == self.n_final_maps:
            return nn_outp.clamp(min=0, max=1)

        # add the 3rd dimension
        nn_outp = torch.cat(
            (nn_outp, self.chan_gen_fcn(nn_outp[:, 1][:, None]) / 2), dim=1)

        # 1. CLAMP
        torch.clamp_(nn_outp[:, :2], min=0, max=1)

        # 4. NORMALIZE it
        nn_outp = nn_outp * 2 - 1
        nn_outp[:, 2] = SurfaceNormalsXTC.SOME_THRESHOLD
        norm_normals_maps = torch.norm(nn_outp, dim=1, keepdim=True)
        norm_normals_maps[norm_normals_maps == 0] = 1
        nn_outp = nn_outp / norm_normals_maps
        nn_outp = (nn_outp + 1) / 2

        return nn_outp

    def postprocess_ensemble_eval(self, nn_outp):
        bs, nch, h, w = nn_outp.shape

        if nch == self.n_final_maps:
            return nn_outp.clamp(min=0, max=1)

        # add the 3rd dimension
        nn_outp = torch.cat(
            (nn_outp, self.chan_gen_fcn(nn_outp[:, 1][:, None]) / 2), dim=1)

        # 1. CLAMP
        torch.clamp_(nn_outp[:, :2], min=0, max=1)

        # 4. NORMALIZE it
        nn_outp = nn_outp * 2 - 1
        nn_outp[:, 2] = SurfaceNormalsXTC.SOME_THRESHOLD
        norm_normals_maps = torch.norm(nn_outp, dim=1, keepdim=True)
        norm_normals_maps[norm_normals_maps == 0] = 1
        nn_outp = nn_outp / norm_normals_maps
        nn_outp = (nn_outp + 1) / 2

        return nn_outp

    def gt_train_transform(edge, gt_inp):
        return gt_inp[:, :2]

    def test_gt(self, loss_fct, pred, target):
        is_nan = target != target

        if is_nan.sum() > 0:
            l_target = target.clone()

            bm = ~is_nan

            l_target[is_nan] = 0
            l_target = l_target * bm
            l_pred = pred * bm
        else:
            l_pred = pred
            l_target = target

        loss = loss_fct(l_pred, l_target)
        return loss

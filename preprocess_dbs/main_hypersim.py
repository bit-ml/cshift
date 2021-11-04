import os
import sys

import cv2
import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# semantic labels - NYU 40
# from https://github.com/ankurhanda/SceneNetv1.0
semantic_labels = [
    'wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door',
    'window', 'bookshelf', 'picture', 'counter', 'blinds', 'desk', 'shelves',
    'curtain', 'dresser', 'pillow', 'mirror', 'floor-mat', 'clothes',
    'ceiling', 'books', 'refrigerator', 'television', 'paper', 'towel',
    'shower-curtain', 'box', 'whiteboard', 'person', 'nightstand', 'toilet',
    'sink', 'lamp', 'bathtub', 'bag', 'other-structure', 'other-furniture',
    'other-prop'
]

sys.path.insert(0,
                os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from experts.depth_expert import DepthModelXTC
from experts.edges_expert import EdgesModel
from experts.grayscale_expert import Grayscale
from experts.halftone_expert import HalftoneModel
from experts.hsv_expert import HSVExpert
from experts.normals_expert import SurfaceNormalsXTC
from experts.sobel_expert import (SobelEdgesExpertSigmaLarge,
                                  SobelEdgesExpertSigmaMedium,
                                  SobelEdgesExpertSigmaSmall)
from experts.superpixel_expert import SuperPixel

# cartoon expert disables eager execution for tf, which is not indicated for edges_expert => should not be used simultaneous
# from experts.cartoon_expert import CartoonWB

# path to store experts results
EXP_OUT_PATH_ = r'/data/multi-domain-graph/datasets/datasets_preproc_exp/hypersim'
# path to store ground truth data
GT_OUT_PATH_ = r'/data/multi-domain-graph/datasets/datasets_preproc_gt/hypersim'
# original dataset info
DATASET_PATH = r'/data/multi-domain-graph/datasets/hypersim/data'

usage_str = 'usage: python main_hypersim.py type split_name exp1 exp2 ...'
#    type                   - [0/1] - 0 create preprocessed gt samples
#                                   - 1 create preprocessed experts samples
#    split_name              - train1/train2/train3/valid/test
#    expi                   - name of the i'th expert / domain
#                           - should be one of the VALID_EXPERTS_NAME / VALID_GT_DOMAINS
#                           - 'all' to run all available experts / domains

VALID_EXPERTS_NAME = [\
    'depth_xtc',
    'normals_xtc',
    'edges_dexined',
    'cartoon_wb',
    'superpixel_fcn',
    'sobel_small',
    'sobel_medium',
    'sobel_large',
    'sem_seg_hrnet' ]

VALID_DOMAINS_NAME = [
    'rgb', 'grayscale', 'hsv', 'halftone_gray', 'depth_n_1', 'normals',
    'sem_seg'
]

COMPUTED_GT_DOMAINS = ['grayscale', 'hsv', 'halftone_gray']

VALID_SPLITS_NAME = ['train1', 'train2', 'test', 'valid']

WORKING_H = 256
WORKING_W = 256

SPLITS_CSV_PATH = r'./hypersim/cshift_hypersim_splits.csv'
DEPTH_ALIGNED_PATH = r'./hypersim/cshift_hypersim_depth_align.npy'

depth_align = np.load(DEPTH_ALIGNED_PATH, allow_pickle=True).item()

RUN_TYPE = 0
EXPERTS_NAME = []
DOMAINS_NAME = []
EXP_OUT_PATH = ''
GT_OUT_PATH = ''
SPLIT_NAME = ''


def check_arguments_without_delete(argv):
    global RUN_TYPE
    global EXPERTS_NAME
    global DOMAINS_NAME
    global EXP_OUT_PATH
    global GT_OUT_PATH
    global SPLIT_NAME
    if len(argv) < 4:
        return 0, 'incorrect usage'

    RUN_TYPE = np.int32(argv[1])
    if not (RUN_TYPE == 0 or RUN_TYPE == 1):
        return 0, 'incorrect run type: %d' % RUN_TYPE

    SPLIT_NAME = argv[2]
    if SPLIT_NAME not in VALID_SPLITS_NAME:
        status = 0
        status_code = 'Split %s is not valid' % SPLIT_NAME
        return status, status_code
    print('SPLIT:', SPLIT_NAME)

    EXP_OUT_PATH = os.path.join(EXP_OUT_PATH_, SPLIT_NAME)
    GT_OUT_PATH = os.path.join(GT_OUT_PATH_, SPLIT_NAME)

    if RUN_TYPE == 0:
        if argv[3] == 'all':
            DOMAINS_NAME = VALID_DOMAINS_NAME
        else:
            potential_domains = argv[3:]
            print("potential_domains", potential_domains)
            print("valid domains", VALID_DOMAINS_NAME)
            DOMAINS_NAME = []

            for i in range(len(potential_domains)):
                dom_name = potential_domains[i]
                if not dom_name in VALID_DOMAINS_NAME:
                    status = 0
                    status_code = 'Domain %s is not valid' % dom_name
                    return status, status_code
                DOMAINS_NAME.append(dom_name)
        print("DOMAINS:", DOMAINS_NAME)
        return 1, ''
    else:
        if argv[3] == 'all':
            EXPERTS_NAME = VALID_EXPERTS_NAME
        else:
            potential_experts = argv[3:]
            print('valid experts ', VALID_EXPERTS_NAME)
            print('potential experts ', potential_experts)
            EXPERTS_NAME = []
            for i in range(len(potential_experts)):
                exp_name = potential_experts[i]
                if not exp_name in VALID_EXPERTS_NAME:
                    status = 0
                    status_code = 'Expert %s is not valid' % exp_name
                    return status, status_code
                EXPERTS_NAME.append(exp_name)
        return 1, ''


def get_task_split_paths(dataset_path, splits_csv_path, split_name, folder_str,
                         task_str, ext):
    df = pd.read_csv(splits_csv_path)
    df = df[df['split'] == split_name]

    paths = []
    for _, row in df.iterrows():
        path = '%s/%s/images/scene_%s_%s/frame.%04d.%s.%s' % (
            dataset_path, row['scene_name'], row['camera_name'], folder_str,
            int(row['frame_id']), task_str, ext)
        paths.append(path)
    return paths


class RGBDataset(Dataset):
    def __init__(self, dataset_path, splits_csv_path, split_name):
        super(RGBDataset, self).__init__()
        self.paths = get_task_split_paths(dataset_path, splits_csv_path,
                                          split_name, 'final_preview',
                                          'tonemap', 'jpg')

    def __getitem__(self, index):
        rgb_info = cv2.imread(self.paths[index])
        rgb_info = cv2.resize(rgb_info, (WORKING_W, WORKING_H),
                              cv2.INTER_CUBIC)
        rgb_info = cv2.cvtColor(rgb_info, cv2.COLOR_BGR2RGB)
        rgb_info = np.moveaxis(rgb_info, -1, 0)
        rgb_info = rgb_info.astype('float32') / 255
        return rgb_info, self.paths[index]

    def __len__(self):
        return len(self.paths)


class RGBDataset_ForExperts(Dataset):
    def __init__(self, dataset_path, splits_csv_path, split_name):
        super(RGBDataset_ForExperts, self).__init__()
        self.paths = get_task_split_paths(dataset_path, splits_csv_path,
                                          split_name, 'final_preview',
                                          'tonemap', 'jpg')

    def __getitem__(self, index):
        rgb_info = cv2.imread(self.paths[index])
        rgb_info = cv2.resize(rgb_info, (WORKING_W, WORKING_H),
                              cv2.INTER_CUBIC)
        rgb_info = cv2.cvtColor(rgb_info, cv2.COLOR_BGR2RGB)
        rgb_info = rgb_info.astype('float32')
        return rgb_info, self.paths[index]

    def __len__(self):
        return len(self.paths)


class TransFct_ScaleMinMax():
    def __init__(self, min_v, max_v):
        self.min_v = min_v
        self.max_v = max_v

    def apply(self, data):
        data = (data - self.min_v) / (self.max_v - self.min_v)
        return data


class TransFct_HistoSpecification():
    def __init__(self, n_bins, cum_data_histo, inv_cum_target_histo):
        self.n_bins = n_bins
        self.cum_data_histo = cum_data_histo
        self.inv_cum_target_histo = inv_cum_target_histo

    def apply(self, data):
        data_ = data * self.n_bins
        data_ = data_.type(torch.int32)
        data_ = self.inv_cum_target_histo[self.cum_data_histo[data_]]
        data_ = data_.astype('float32')
        data_ = data_ / self.n_bins
        return data_


class TransFct_DepthExp():
    def __init__(self, min_v, max_v, n_bins, cum_data_histo,
                 inv_cum_target_histo):
        self.min_v = min_v
        self.max_v = max_v
        self.n_bins = n_bins
        self.cum_data_histo = cum_data_histo
        self.inv_cum_target_histo = inv_cum_target_histo

    def apply(self, data):
        data = (data - self.min_v) / (self.max_v - self.min_v)
        data_ = data * self.n_bins
        data_ = data_.astype('int32')
        data_ = self.inv_cum_target_histo[self.cum_data_histo[data_]]
        data_ = data_.astype('float32')
        data_ = data_ / self.n_bins
        data_ = data_.astype('float32')
        data = data_
        return data


class DepthDataset(Dataset):
    def __init__(self, dataset_path, splits_csv_path, split_name):
        super(DepthDataset, self).__init__()
        self.paths = get_task_split_paths(dataset_path, splits_csv_path,
                                          split_name, 'geometry_hdf5',
                                          'depth_meters', 'hdf5')

        self.scale_min_max_fct = TransFct_ScaleMinMax(depth_align['gt_min'],
                                                      depth_align['gt_max'])
        self.histo_specification = TransFct_HistoSpecification(
            depth_align['gt_n_bins'], depth_align['gt_cum_data_histo'],
            depth_align['gt_inv_cum_target_histo'])

    def __getitem__(self, index):
        depth_file = h5py.File(self.paths[index], "r")
        depth_info = np.array(depth_file.get('dataset')).astype('float32')
        depth_info = torch.from_numpy(depth_info).unsqueeze(0)
        depth_info = torch.nn.functional.interpolate(depth_info[None],
                                                     (WORKING_H, WORKING_W))[0]
        nan_mask = depth_info != depth_info
        depth_info[nan_mask] = 0  # get rid of nan values
        depth_info = self.scale_min_max_fct.apply(depth_info)
        depth_info = self.histo_specification.apply(depth_info)
        depth_info[nan_mask] = float("nan")
        return depth_info, self.paths[index]

    def __len__(self):
        return len(self.paths)


class NormalsDataset(Dataset):
    def __init__(self, dataset_path, splits_csv_path, split_name):
        super(NormalsDataset, self).__init__()
        self.paths = get_task_split_paths(dataset_path, splits_csv_path,
                                          split_name, 'geometry_hdf5',
                                          'normal_cam', 'hdf5')

    def __getitem__(self, index):
        normal_file = h5py.File(self.paths[index], "r")
        normal_info = np.array(normal_file.get('dataset')).astype('float32')
        normal_info = torch.from_numpy(normal_info).permute(2, 0, 1)
        normal_info = torch.nn.functional.interpolate(
            normal_info[None], (WORKING_H, WORKING_W))[0]

        nan_mask = normal_info != normal_info

        normal_info[1, :, :] = normal_info[1, :, :] * (-1)
        normal_info[2, :, :] = normal_info[2, :, :] * (-1)

        normal_info[
            2, :, :] = experts.normals_expert.SurfaceNormalsXTC.SOME_THRESHOLD

        norm = torch.norm(normal_info, dim=0, keepdim=True)
        norm[norm == 0] = 1
        normal_info = normal_info / norm

        normal_info = (normal_info + 1) / 2
        normal_info[nan_mask] = float("nan")
        return normal_info, self.paths[index]

    def __len__(self):
        return len(self.paths)


class SemanticSegDataset(Dataset):
    def __init__(self, dataset_path, splits_csv_path, split_name):
        super(SemanticSegDataset, self).__init__()
        self.paths = get_task_split_paths(dataset_path, splits_csv_path,
                                          split_name, 'geometry_hdf5',
                                          'semantic', 'hdf5')

    def __getitem__(self, index):
        semantic_file = h5py.File(self.paths[index], "r")
        semantic_info = np.array(semantic_file.get('dataset'))
        semantic_info[semantic_info == -1] = 0
        semantic_info = np.uint8(semantic_info)

        semantic_info = cv2.resize(semantic_info, (WORKING_W, WORKING_H),
                                   cv2.INTER_NEAREST)
        semantic_info = semantic_info.astype('float32')
        semantic_info[semantic_info == -1] = 0
        semantic_info = semantic_info[None]
        return semantic_info, self.paths[index]

    def __len__(self):
        return len(self.paths)


def get_expert(exp_name):
    if exp_name == 'depth_xtc':
        return DepthModelXTC(full_expert=True)
    elif exp_name == 'normals_xtc':
        return SurfaceNormalsXTC(dataset_name='hypersim', full_expert=True)
    elif exp_name == 'edges_dexined':
        return EdgesModel(full_expert=True)
    elif exp_name == 'cartoon_wb':
        return CartoonWB(full_expert=True)
    elif exp_name == 'sobel_small':
        return SobelEdgesExpertSigmaSmall()
    elif exp_name == 'sobel_medium':
        return SobelEdgesExpertSigmaMedium()
    elif exp_name == 'sobel_large':
        return SobelEdgesExpertSigmaLarge()
    elif exp_name == 'superpixel_fcn':
        return SuperPixel()
    elif exp_name == 'sem_seg_hrnet':
        import experts.semantic_segmentation_expert
        return experts.semantic_segmentation_expert.SSegHRNet(
            dataset_name='hypersim', full_expert=True)
    return None


def add_normals_process(data):
    data = np.clip(data, 0, 1)
    data = data * 2 - 1
    data[:, 2] = experts.normals_expert.SurfaceNormalsXTC.SOME_THRESHOLD
    norm_data = np.linalg.norm(data, axis=1, keepdims=True)
    norm_data[norm_data == 0] = 1
    data = data / norm_data
    data = (data + 1) / 2
    return data


def get_exp_results():
    print("get experts ", EXPERTS_NAME)

    if 'depth_xtc' in EXPERTS_NAME:
        depth_exp_trans_fct = TransFct_DepthExp(
            depth_align['exp_min'], depth_align['exp_max'],
            depth_align['exp_n_bins'], depth_align['exp_cum_data_histo'],
            depth_align['exp_inv_cum_target_histo'])

    dataset = RGBDataset_ForExperts(DATASET_PATH, SPLITS_CSV_PATH, SPLIT_NAME)
    dataloader = DataLoader(dataset,
                            batch_size=30,
                            shuffle=False,
                            num_workers=20)
    for exp_name in EXPERTS_NAME:
        exp_out_path = os.path.join(EXP_OUT_PATH, exp_name)
        os.makedirs(exp_out_path, exist_ok=True)
        expert = get_expert(exp_name)
        process_fct = expert.apply_expert_batch
        if exp_name == 'depth_xtc':
            add_process_fct = depth_exp_trans_fct.apply
        elif exp_name == 'normals_xtc':
            add_process_fct = add_normals_process
        else:
            add_process_fct = lambda x: x
        file_idx = 0

        for batch in tqdm(dataloader):
            exp_info, paths = batch
            exp_info = process_fct(exp_info)
            exp_info = add_process_fct(exp_info)
            for i in range(exp_info.shape[0]):
                exp_info_ = exp_info[i]
                exp_info_ = np.array(exp_info_)
                np.save(os.path.join(exp_out_path, '%08d.npy' % file_idx),
                        exp_info_)
                file_idx += 1
    return


def get_expert_gt(dom_name):
    if dom_name == 'grayscale':
        return Grayscale(full_expert=True)
    elif dom_name == 'hsv':
        return HSVExpert(full_expert=True)
    elif dom_name == 'halftone_gray':
        return HalftoneModel(full_expert=True, style=0)
    return None


def get_dataset_type(dom_name):
    if dom_name in COMPUTED_GT_DOMAINS:
        return RGBDataset_ForExperts
    elif dom_name == 'rgb':
        return RGBDataset
    elif dom_name == 'depth_n_1':
        return DepthDataset
    elif dom_name == 'normals':
        return NormalsDataset
    elif dom_name == 'sem_seg':
        return SemanticSegDataset
    return None


def get_gt_domains():
    print("get_gt_domains", DOMAINS_NAME)

    for dom in DOMAINS_NAME:
        dom_out_path = os.path.join(GT_OUT_PATH, dom)
        os.makedirs(dom_out_path, exist_ok=True)

        if dom in COMPUTED_GT_DOMAINS:
            expert = get_expert_gt(dom)
            process_fct = expert.apply_expert_batch
        else:
            process_fct = lambda x: x

        task_dataset_cls = get_dataset_type(dom)

        dataset = task_dataset_cls(DATASET_PATH, SPLITS_CSV_PATH, SPLIT_NAME)
        dataloader = DataLoader(dataset,
                                batch_size=100,
                                shuffle=False,
                                num_workers=20)

        file_idx = 0
        for batch in tqdm(dataloader):
            dom_info, paths = batch
            dom_info = process_fct(dom_info)

            for i in range(dom_info.shape[0]):
                dom_info_ = dom_info[i]
                dom_info_ = np.array(dom_info_)
                np.save(os.path.join(dom_out_path, '%08d.npy' % file_idx),
                        dom_info_)
                file_idx += 1


if __name__ == "__main__":
    status, status_code = check_arguments_without_delete(sys.argv)
    if status == 0:
        sys.exit(status_code + '\n' + usage_str)

    if RUN_TYPE == 0:
        get_gt_domains()
    else:
        get_exp_results()

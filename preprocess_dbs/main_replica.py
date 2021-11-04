import glob
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0,
                os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import experts.depth_expert
import experts.grayscale_expert
import experts.halftone_expert
import experts.hsv_expert
import experts.normals_expert
import experts.rgb_expert

WORKING_H = 256
WORKING_W = 256

# dataset domain names
VALID_ORIG_GT_DOMAINS = [
    'rgb', 'depth', 'normals', "halftone_gray", "grayscale", "hsv"
]

# our internal domain names
VALID_GT_DOMAINS = [\
    'rgb',
    'depth_n_1',
    'normals',
    'halftone_gray',
    'grayscale',
    'hsv'\
]

VALID_EXPERTS_NAME = [\
    'depth_xtc',
    'normals_xtc',
    'sem_seg_hrnet',
    'superpixel_fcn',
    'sobel_small',
    'sobel_medium',
    'sobel_large',
    'cartoon_wb',
    'edges_dexined']
VALID_SPLITS_NAME = ["valid", "test", "train1", "train2"]

RUN_TYPE = []
EXPERTS_NAME = []
ORIG_DOMAINS = []
DOMAINS = []
SPLIT_NAME = ''

usage_str = 'usage: python main_taskonomy.py type split-name exp1 exp2 ...'
#    type                   - [0/1] - 0 create preprocessed gt samples
#                                   - 1 create preprocessed experts samples
#    expi                   - name of the i'th expert / domain
#                           - should be one of the VALID_EXPERTS_NAME / VALID_GT_DOMAINS
#                           - 'all' to run all available experts / domains
#    split-name             - should be one of the VALID_SPLITS_NAME

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

REPLICA_PROC_NAME = "replica"
REPLICA_RAW_NAME = "replica_raw"

DEPTH_ALIGNED_PATH = './replica_generator/cshift_%s_depth_align.npy' % REPLICA_PROC_NAME
depth_align = np.load(DEPTH_ALIGNED_PATH, allow_pickle=True).item()


def check_arguments_without_delete(argv):
    global RUN_TYPE
    global EXPERTS_NAME
    global ORIG_DOMAINS
    global DOMAINS
    global MAIN_DB_PATH
    global MAIN_GT_OUT_PATH
    global MAIN_EXP_OUT_PATH
    global SPLIT_NAME

    if len(argv) < 4:
        return 0, 'incorrect usage'

    RUN_TYPE = np.int32(argv[1])
    if not (RUN_TYPE == 0 or RUN_TYPE == 1 or RUN_TYPE == 2 or RUN_TYPE == 3):
        return 0, 'incorrect run type: %d' % RUN_TYPE

    split_name = argv[2]
    if split_name not in VALID_SPLITS_NAME:
        status = 0
        status_code = 'Split %s is not valid. Valid ones are: %s' % (
            split_name, VALID_SPLITS_NAME)
        return status, status_code
    SPLIT_NAME = split_name

    MAIN_DB_PATH = r'/data/multi-domain-graph/datasets/%s/%s' % (
        REPLICA_RAW_NAME, split_name)
    MAIN_GT_OUT_PATH = r'/data/multi-domain-graph/datasets/datasets_preproc_gt/%s/%s' % (
        REPLICA_PROC_NAME, split_name)
    MAIN_EXP_OUT_PATH = r'/data/multi-domain-graph/datasets/datasets_preproc_exp/%s/%s' % (
        REPLICA_PROC_NAME, split_name)

    os.system("mkdir -p %s" % MAIN_GT_OUT_PATH)
    os.system("mkdir -p %s" % MAIN_EXP_OUT_PATH)

    if RUN_TYPE == 0:
        if argv[3] == 'all':
            ORIG_DOMAINS = []
            DOMAINS = []
            for doms in zip(VALID_ORIG_GT_DOMAINS, VALID_GT_DOMAINS):
                orig_dom_name, dom_name = doms
                ORIG_DOMAINS.append(orig_dom_name)
                DOMAINS.append(dom_name)
        else:
            potential_domains = argv[3:]
            print("potential_domains", potential_domains)
            print("VALID_GT_DOMAINS", VALID_GT_DOMAINS)
            ORIG_DOMAINS = []
            DOMAINS = []
            for i in range(len(potential_domains)):
                dom_name = potential_domains[i]
                if not dom_name in VALID_GT_DOMAINS:
                    status = 0
                    status_code = 'Domain %s is not valid' % dom_name
                    return status, status_code
                orig_dom_name = VALID_ORIG_GT_DOMAINS[VALID_GT_DOMAINS.index(
                    dom_name)]

                ORIG_DOMAINS.append(orig_dom_name)
                DOMAINS.append(dom_name)
        print("ORIG_DOMAINS", ORIG_DOMAINS)
        return 1, ''
    elif RUN_TYPE == 1:
        if argv[3] == 'all':
            EXPERTS_NAME = []
            for exp_name in VALID_EXPERTS_NAME:
                EXPERTS_NAME.append(exp_name)
        else:
            potential_experts = argv[3:]
            EXPERTS_NAME = []
            for i in range(len(potential_experts)):
                exp_name = potential_experts[i]
                if not exp_name in VALID_EXPERTS_NAME:
                    status = 0
                    status_code = 'Expert %s is not valid' % exp_name
                    return status, status_code
                EXPERTS_NAME.append(exp_name)
        return 1, ''
    else:
        return 1, ''


def get_expert(exp_name):
    if exp_name == 'halftone_gray':
        return experts.halftone_expert.HalftoneModel(full_expert=True, style=0)
    elif exp_name == 'depth_xtc':
        import experts.depth_expert
        return experts.depth_expert.DepthModelXTC(full_expert=True)
    elif exp_name == 'edges_dexined':
        import experts.edges_expert
        return experts.edges_expert.EdgesModel(full_expert=True)
    elif exp_name == 'normals_xtc':
        global SURFNORM_KERNEL
        SURFNORM_KERNEL = torch.from_numpy(
            np.array([
                [[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
                [[1, 0, -1], [2, 0, -2], [1, 0, -1]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            ]))[:, np.newaxis, ...].to(dtype=torch.float32, device=device)

        return experts.normals_expert.SurfaceNormalsXTC(dataset_name="replica",
                                                        full_expert=True)
    elif exp_name == 'rgb':
        return experts.rgb_expert.RGBModel(full_expert=True)
    elif exp_name == 'sem_seg_hrnet':
        import experts.semantic_segmentation_expert
        return experts.semantic_segmentation_expert.SSegHRNet(
            dataset_name="replica", full_expert=True)
    elif exp_name == 'grayscale':
        return experts.grayscale_expert.Grayscale(full_expert=True)
    elif exp_name == 'hsv':
        return experts.hsv_expert.HSVExpert(full_expert=True)
    elif exp_name == 'cartoon_wb':
        import experts.cartoon_expert
        return experts.cartoon_expert.CartoonWB(full_expert=True)
    elif exp_name == 'sobel_small':
        import experts.sobel_expert
        return experts.sobel_expert.SobelEdgesExpertSigmaSmall()
    elif exp_name == 'sobel_medium':
        import experts.sobel_expert
        return experts.sobel_expert.SobelEdgesExpertSigmaMedium()
    elif exp_name == 'sobel_large':
        import experts.sobel_expert
        return experts.sobel_expert.SobelEdgesExpertSigmaLarge()
    elif exp_name == 'superpixel_fcn':
        import experts.superpixel_expert
        return experts.superpixel_expert.SuperPixel()


def depth_to_surface_normals(depth, surfnorm_scalar=256):
    with torch.no_grad():
        surface_normals = F.conv2d(depth,
                                   surfnorm_scalar * SURFNORM_KERNEL,
                                   padding=1)
        surface_normals[:, 2, ...] = 1
        norm = surface_normals.norm(dim=1, keepdim=True)
        surface_normals = surface_normals / norm

    return surface_normals


def process_gt_from_expert(domain_name):
    get_exp_results(MAIN_GT_OUT_PATH, experts_name=[domain_name])

    # link it in the experts
    to_unlink = os.path.join(MAIN_EXP_OUT_PATH, domain_name)
    os.system("unlink %s" % (to_unlink))

    from_path = os.path.join(MAIN_GT_OUT_PATH, domain_name)
    os.system("ln -s %s %s" % (from_path, MAIN_EXP_OUT_PATH))


def process_rgb(in_path, out_path):
    os.makedirs(out_path, exist_ok=True)
    os.system("cp -r '%s/'* '%s'" % (in_path, out_path))

    # link it in the experts
    to_unlink = os.path.join(MAIN_EXP_OUT_PATH, "rgb")
    os.system("unlink %s" % (to_unlink))
    os.system("ln -s %s %s" % (out_path, MAIN_EXP_OUT_PATH))


class TransFct_ScaleMinMax():
    def __init__(self, min_v, max_v):
        self.min_v = min_v
        self.max_v = max_v

    def apply(self, data):
        data = (data - self.min_v) / (self.max_v - self.min_v)
        return data


class TransFct_HistoHalfClamp():
    def __init__(self, th_path):
        self.th_95 = np.load(th_path)

    def apply(self, data):
        data = data / self.th_95
        return data


class TransFct_HistoSpecification():
    def __init__(self, n_bins, cum_data_histo, inv_cum_target_histo):
        self.n_bins = n_bins
        self.cum_data_histo = cum_data_histo
        self.inv_cum_target_histo = inv_cum_target_histo

    def apply(self, data):

        data_ = data * self.n_bins
        data_ = data_.astype('int32')
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


class GT_DepthDataset(Dataset):
    def __init__(self, depth_path, split_name):
        super(GT_DepthDataset, self).__init__()
        if split_name == 'valid':
            split_name = 'val'
        glob_pattern = '%s/*.npy' % (depth_path)
        self.depth_paths = sorted(glob.glob(glob_pattern))

        self.scale_min_max_fct = TransFct_ScaleMinMax(depth_align['gt_min'],
                                                      depth_align['gt_max'])
        self.histo_specification = TransFct_HistoSpecification(
            depth_align['gt_n_bins'], depth_align['gt_cum_data_histo'],
            depth_align['gt_inv_cum_target_histo'])

    def __getitem__(self, index):
        depth = np.load(self.depth_paths[index])
        bm = depth == 0

        depth = self.scale_min_max_fct.apply(depth)
        depth = self.histo_specification.apply(depth)
        depth[bm] = float("nan")

        depth = depth[None]
        return depth

    def __len__(self):
        return len(self.depth_paths)


def process_depth(in_path, out_path):
    os.makedirs(out_path, exist_ok=True)

    depth_dataset = GT_DepthDataset(in_path, SPLIT_NAME)
    dataloader = DataLoader(depth_dataset,
                            batch_size=150,
                            shuffle=False,
                            num_workers=20,
                            drop_last=False)
    files = os.listdir(in_path)
    files.sort()

    file_idx = 0
    for batch in tqdm(dataloader):
        depth = batch
        for i in range(depth.shape[0]):
            depth_ = depth[i]
            depth_ = np.array(depth_)
            np.save(os.path.join(out_path, '%08d.npy' % file_idx), depth_)
            file_idx += 1


def process_surface_normals(main_db_path, out_path):
    os.makedirs(out_path, exist_ok=True)
    depth_full_path = os.path.join(main_db_path, "depth_n")

    batch_size = 500
    dataset = DatasetDepth(depth_full_path)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             drop_last=False,
                                             num_workers=8)
    with torch.no_grad():
        for batch_idx, (depth_frames, indexes) in enumerate(tqdm(dataloader)):
            # adjust depth
            depth_frames = 1 - depth_frames
            depth_frames = depth_frames.to(device)

            gt_maps = depth_to_surface_normals(depth_frames)
            gt_maps = (gt_maps + 1.) / 2.

            # permute it to match the normals expert
            permutation = [1, 0, 2]
            gt_maps = gt_maps[:, permutation]

            # 4. NORMALIZE it
            gt_maps = gt_maps * 2 - 1
            gt_maps[:,
                    2] = experts.normals_expert.SurfaceNormalsXTC.SOME_THRESHOLD
            norm_normals_maps = torch.norm(gt_maps, dim=1, keepdim=True)
            norm_normals_maps[norm_normals_maps == 0] = 1
            gt_maps = gt_maps / norm_normals_maps
            gt_maps = (gt_maps + 1) / 2

            # SAVE Normals npy
            for sample in zip(gt_maps, indexes):
                normals_img, sample_idx = sample

                normals_img_path = os.path.join(out_path,
                                                '%08d.npy' % sample_idx)
                # TODO: save all batch, in smtg with workers like get_item
                np.save(normals_img_path, normals_img.data.cpu().numpy())


def get_gt_domains():
    print("get_gt_domains", ORIG_DOMAINS)
    for doms in zip(ORIG_DOMAINS, DOMAINS):
        orig_dom_name, dom_name = doms

        in_path = os.path.join(MAIN_DB_PATH, orig_dom_name)
        out_path = os.path.join(MAIN_GT_OUT_PATH, dom_name)

        if orig_dom_name == 'rgb':
            process_rgb(in_path, out_path)
        elif orig_dom_name == 'depth':
            process_depth(in_path, out_path)
        elif orig_dom_name == 'normals':
            process_surface_normals(MAIN_GT_OUT_PATH, out_path)
        elif orig_dom_name in ['grayscale', 'halftone_gray', 'hsv']:
            process_gt_from_expert(orig_dom_name)


class DatasetDepth(Dataset):
    def __init__(self, depth_paths):
        super(DatasetDepth, self).__init__()

        filenames = os.listdir(depth_paths)
        filenames.sort()
        self.depth_paths = []
        for filename in filenames:
            self.depth_paths.append(os.path.join(depth_paths, filename))

    def __getitem__(self, index):
        depth_npy = np.load(self.depth_paths[index])
        index_of_file = int(self.depth_paths[index].split("/")[-1].replace(
            ".npy", ""))
        return depth_npy, index_of_file

    def __len__(self):
        return len(self.depth_paths)


class Dataset_ImgLevel(Dataset):
    def __init__(self, rgbs_path):
        super(Dataset_ImgLevel, self).__init__()

        filenames = os.listdir(rgbs_path)
        filenames.sort()
        self.rgbs_path = []
        for filename in filenames:
            self.rgbs_path.append(os.path.join(rgbs_path, filename))

    def __getitem__(self, index):
        rgb = np.load(self.rgbs_path[index])
        index_of_file = int(self.rgbs_path[index].split("/")[-1].replace(
            ".npy", ""))
        return rgb, index_of_file

    def __len__(self):
        return len(self.rgbs_path)


def get_exp_results(main_exp_out_path, experts_name):
    if 'depth_xtc' in experts_name:
        depth_exp_trans_fct = TransFct_DepthExp(
            depth_align['exp_min'], depth_align['exp_max'],
            depth_align['exp_n_bins'], depth_align['exp_cum_data_histo'],
            depth_align['exp_inv_cum_target_histo'])

    with torch.no_grad():
        rgbs_path = os.path.join(MAIN_DB_PATH, 'rgb')
        batch_size = 100

        dataset = Dataset_ImgLevel(rgbs_path)
        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=batch_size,
                                                 shuffle=False,
                                                 drop_last=False,
                                                 num_workers=8)

        for exp_name in experts_name:
            if exp_name in ["sem_seg_hrnet", "normals_xtc"]:
                dataloader = torch.utils.data.DataLoader(dataset,
                                                         batch_size=80,
                                                         shuffle=False,
                                                         drop_last=False,
                                                         num_workers=8)
            print('EXPERT: %20s' % exp_name)
            expert = get_expert(exp_name)

            if exp_name == 'depth_xtc':
                #post_process_fct = post_process_depth_xtc_fct
                post_process_fct = depth_exp_trans_fct.apply
            else:
                post_process_fct = lambda x: x
            exp_out_path = os.path.join(main_exp_out_path, exp_name)
            os.makedirs(exp_out_path, exist_ok=True)

            for batch_idx, (frames, indexes) in enumerate(tqdm(dataloader)):
                # skip fast (eg. for missing depth 00161500.npy)
                # if indexes[-1] < 161499:
                #     continue

                frames = frames.permute(0, 2, 3, 1) * 255.
                results = expert.apply_expert_batch(frames)
                results = post_process_fct(results)

                for sample in zip(results, indexes):
                    expert_res, sample_idx = sample

                    out_path = os.path.join(exp_out_path,
                                            '%08d.npy' % sample_idx)

                    np.save(out_path, expert_res)


if __name__ == "__main__":
    status, status_code = check_arguments_without_delete(sys.argv)
    if status == 0:
        sys.exit(status_code + '\n' + usage_str)
    print(MAIN_DB_PATH)

    if RUN_TYPE == 0:
        get_gt_domains()
    elif RUN_TYPE == 1:
        get_exp_results(MAIN_EXP_OUT_PATH, EXPERTS_NAME)

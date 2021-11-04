import glob
import os
import pathlib

import numpy as np
from torch.utils.data import Dataset

experts_using_gt_in_second_iter = ['rgb', 'hsv', 'grayscale', 'halftone_gray']


def load_glob_with_cache_multiple_patterns(cache_file, glob_paths):
    if not os.path.exists(cache_file):
        all_paths = sorted(glob.glob(glob_paths[0]))
        for i in np.arange(1, len(glob_paths)):
            all_paths = all_paths + glob.glob(glob_paths[i])
        all_paths = sorted(all_paths)

        save_folder = os.path.dirname(cache_file)
        if not os.path.exists(save_folder):
            pathlib.Path(save_folder).mkdir(parents=True, exist_ok=True)
        np.save(cache_file, all_paths)
    else:
        all_paths = np.load(cache_file)
    return all_paths


def get_paths_for_idx_and_split(config, split_str, src_expert, dst_expert):
    src_path = config.get('PathsIter',
                          'ITER_%s_PATH' % (split_str)).split('\n')
    dst_path = src_path

    first_k = config.getint('PathsIter', 'ITER_%s_FIRST_K' % (split_str))
    if split_str == 'TEST':
        gt_dst_path = config.get('PathsIter', 'ITER_%s_GT_DST_PATH' %
                                 (split_str)).split('\n')
    else:
        gt_dst_path = None

    if src_expert.identifier in experts_using_gt_in_second_iter:
        src_path = config.get('PathsIter',
                              'ITER_%s_GT_DST_PATH' % (split_str)).split('\n')
    if dst_expert.identifier in experts_using_gt_in_second_iter:
        dst_path = config.get('PathsIter',
                              'ITER_%s_GT_DST_PATH' % (split_str)).split('\n')
    return src_path, dst_path, first_k, gt_dst_path


def get_glob_paths(path, identifier):
    glob_paths = []
    for i in range(len(path)):
        glob_paths_ = ["%s/%s/*.npy" % (path[i], identifier)]
        glob_paths = glob_paths + glob_paths_
    return glob_paths


class ImageLevelDataset(Dataset):
    def __init__(self, src_expert, dst_expert, config, split_str):
        """
            src_expert
            dst_expert 
            config 
            iter_idx - current iteration index 
            split_str - desired split ('TRAIN', 'VALID' or 'TEST')
            for_next_iter - if we load a dataset for next iter (load it in order to save ensembles of current iter)
            for_next_iter_idx_subset - if we load a dataset for next iter, we can load per subset s.t. we can save it in corresponding subset dst 
        """
        super(ImageLevelDataset, self).__init__()
        self.src_expert = src_expert
        self.dst_expert = dst_expert

        src_path, dst_path, first_k, gt_dst_path = get_paths_for_idx_and_split(
            config, split_str, src_expert, dst_expert)

        if first_k == 0:
            self.src_paths = []
            self.dst_path = []
            self.gt_dst_paths = []
            return

        paths_str = '_'.join(src_path).replace("/", "__")

        tag = 'split_%s_nPaths_%d_%s' % (split_str, len(src_path), paths_str)

        CACHE_NAME = config.get('General', 'CACHE_NAME')
        use_cache = config.getboolean('General', 'use_cache')
        if not use_cache:
            datetime_str = config.get('Run id', 'datetime')
            CACHE_NAME += "/" + datetime_str

        cache_src = "%s/src_%s_%s.npy" % (CACHE_NAME, tag,
                                          self.src_expert.identifier)
        glob_paths_srcs = get_glob_paths(src_path, self.src_expert.identifier)
        self.src_paths = load_glob_with_cache_multiple_patterns(
            cache_src, glob_paths_srcs)
        self.src_paths = self.src_paths[:len(self.src_paths) if first_k ==
                                        -1 else min(first_k, len(self.src_paths
                                                                 ))]

        cache_dst = "%s/dst_%s_%s.npy" % (CACHE_NAME, tag,
                                          self.dst_expert.identifier)
        glob_paths_dsts = get_glob_paths(dst_path, self.dst_expert.identifier)
        self.dst_paths = load_glob_with_cache_multiple_patterns(
            cache_dst, glob_paths_dsts)
        self.dst_paths = self.dst_paths[:len(self.dst_paths) if first_k ==
                                        -1 else min(first_k, len(self.dst_paths
                                                                 ))]
        assert (len(self.src_paths) == len(self.dst_paths))

        if gt_dst_path:
            cache_gt_dst = "%s/gt_dst_%s_%s.npy" % (
                CACHE_NAME, tag, self.dst_expert.domain_name)
            glob_paths_gt_dsts = get_glob_paths(gt_dst_path,
                                                self.dst_expert.domain_name)
            self.gt_dst_paths = load_glob_with_cache_multiple_patterns(
                cache_gt_dst, glob_paths_gt_dsts)
            self.gt_dst_paths = self.gt_dst_paths[:len(
                self.gt_dst_paths
            ) if first_k == -1 else min(first_k, len(self.gt_dst_paths))]
            assert (len(self.src_paths) == len(self.gt_dst_paths))
        else:
            self.gt_dst_paths = []

    def __getitem__(self, index):
        src_data = np.load(self.src_paths[index]).astype(np.float32)
        dst_data = np.load(self.dst_paths[index])

        if len(self.gt_dst_paths) > 0:
            gt_dst_data = np.load(self.gt_dst_paths[index])
            return src_data, dst_data, gt_dst_data
        else:
            return src_data, dst_data

    def __len__(self):
        return len(self.src_paths)


class ImageStoreDataset(ImageLevelDataset):
    def __init__(self, src_expert, dst_expert, config, split_str):
        super(ImageStoreDataset, self).__init__(src_expert, dst_expert, config,
                                                split_str)

    def __getitem__(self, index):
        src_data = np.load(self.src_paths[index]).astype(np.float32)
        dst_data = np.load(self.dst_paths[index])

        return src_data, dst_data

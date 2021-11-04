#
# For licensing see accompanying LICENSE.txt file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#
import argparse
import os
import shutil
import glob

# usage python dataset_download_images.py --downloads_dir [zip files dir] --decompress_dir [final files dir] --delete_archive_after_decompress

non_required_color_data = [
    'diff', 'diffuse_illumination', 'diffuse_reflectance', 'gamma',
    'lambertian', 'non_lambertian', 'residual'
]
non_required_color_hdf5_data = [
    'diff', 'diffuse_illumination', 'diffuse_reflectance', 'gamma',
    'lambertian', 'non_lambertian', 'residual'
]
non_required_geo_data = [
    'normal_bump_cam', 'normal_bump_world', 'normal_world', 'position',
    'render_entity_id', 'tex_coord'
]

parser = argparse.ArgumentParser()
parser.add_argument("--downloads_dir", required=True)
parser.add_argument("--decompress_dir")
parser.add_argument("--delete_archive_after_decompress", action="store_true")
args = parser.parse_args()

print("[HYPERSIM: DATASET_DOWNLOAD_IMAGES] Begin...")

if not os.path.exists(args.downloads_dir): os.makedirs(args.downloads_dir)

if args.decompress_dir is not None:
    if not os.path.exists(args.decompress_dir):
        os.makedirs(args.decompress_dir)


def remove_all(paths):
    for path in paths:
        if os.path.exists(path):
            if os.path.isfile(path):
                os.remove(path)
            else:
                shutil.rmtree(path)


def download(url):
    download_name = os.path.basename(url)
    download_file = os.path.join(args.downloads_dir, download_name)

    cmd = "curl " + url + " --output " + download_file
    print("")
    print(cmd)
    print("")
    retval = os.system(cmd)
    assert retval == 0

    if args.decompress_dir is not None:
        cmd = "unzip " + download_file + " -d " + args.decompress_dir
        print("")
        print(cmd)
        print("")
        retval = os.system(cmd)
        assert retval == 0
        if args.delete_archive_after_decompress:
            cmd = "rm " + download_file
            print("")
            print(cmd)
            print("")
            retval = os.system(cmd)
            assert retval == 0

    pos = url.find('ai_')
    scene_name = url[pos:-4]

    scene_path = os.path.join(args.decompress_dir, scene_name)
    # remove dataset details
    details_path = os.path.join(scene_path, '_detail')
    if os.path.exists(details_path):
        shutil.rmtree(details_path)
    images_path = os.path.join(scene_path, 'images')

    # remove non required geometry preview data
    glob_path = "%s/*geometry_preview" % (images_path)
    remove_all(glob.glob(glob_path))

    # remove non required final hdf5 data
    glob_path = "%s/*final_hdf5" % (images_path)
    all_final_preview = sorted(glob.glob(glob_path))
    for cam_path in all_final_preview:
        for str_color in non_required_color_hdf5_data:
            glob_path_ = "%s/*%s.hdf5" % (cam_path, str_color)
            remove_all(glob.glob(glob_path_))
    #remove_all(glob.glob(glob_path))

    # remove non required final preview data
    glob_path = "%s/*final_preview" % (images_path)
    all_final_preview = sorted(glob.glob(glob_path))
    for cam_path in all_final_preview:
        for str_color in non_required_color_data:
            glob_path_ = "%s/*%s.jpg" % (cam_path, str_color)
            remove_all(glob.glob(glob_path_))
    # remove non required geometry hdf5 data
    glob_path = "%s/*geometry_hdf5" % (images_path)
    all_geometry_hdf5 = sorted(glob.glob(glob_path))
    for cam_path in all_geometry_hdf5:
        for str_geo in non_required_geo_data:
            glob_path_ = "%s/*%s.hdf5" % (cam_path, str_geo)
            remove_all(glob.glob(glob_path_))


urls_to_download = [
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_001_001.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_001_002.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_001_003.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_001_004.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_001_005.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_001_006.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_001_007.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_001_008.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_001_009.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_001_010.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_002_001.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_002_002.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_002_003.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_002_004.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_002_005.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_002_006.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_002_007.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_002_008.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_002_009.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_002_010.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_003_001.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_003_002.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_003_004.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_003_005.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_003_006.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_003_007.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_003_008.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_003_009.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_003_010.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_004_001.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_004_002.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_004_003.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_004_004.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_004_005.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_004_006.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_004_007.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_004_008.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_004_009.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_004_010.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_005_001.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_005_003.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_005_004.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_005_005.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_005_006.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_005_007.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_005_008.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_005_009.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_005_010.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_006_001.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_006_002.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_006_003.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_006_004.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_006_006.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_006_007.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_006_008.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_006_009.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_006_010.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_007_001.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_007_002.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_007_004.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_007_005.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_007_006.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_007_007.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_007_008.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_007_009.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_007_010.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_008_001.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_008_002.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_008_003.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_008_004.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_008_005.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_008_006.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_008_007.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_008_008.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_008_009.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_008_010.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_009_001.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_009_002.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_009_003.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_009_004.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_009_005.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_009_006.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_009_007.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_009_008.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_009_009.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_010_001.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_010_002.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_010_003.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_010_004.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_010_005.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_010_006.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_010_007.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_010_008.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_010_009.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_011_001.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_011_003.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_011_004.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_011_005.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_011_006.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_011_007.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_011_008.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_011_009.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_011_010.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_012_001.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_012_004.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_012_005.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_012_007.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_012_009.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_012_010.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_013_001.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_013_002.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_013_003.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_013_004.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_013_007.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_013_009.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_013_010.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_014_003.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_014_006.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_014_010.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_015_001.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_015_003.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_015_004.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_015_005.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_015_006.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_015_007.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_015_008.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_015_009.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_015_010.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_016_001.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_016_002.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_016_003.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_016_004.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_016_005.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_016_006.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_016_007.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_016_009.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_016_010.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_017_001.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_017_002.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_017_003.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_017_004.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_017_005.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_017_006.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_017_007.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_017_008.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_017_009.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_017_010.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_018_001.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_018_002.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_018_003.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_018_004.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_018_005.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_018_006.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_018_007.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_018_008.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_018_009.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_018_010.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_019_001.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_019_002.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_019_003.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_019_004.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_019_006.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_019_007.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_019_008.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_019_009.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_021_001.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_021_002.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_021_003.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_021_007.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_021_008.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_021_009.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_021_010.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_022_001.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_022_002.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_022_003.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_022_004.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_022_005.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_022_006.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_022_007.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_022_009.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_022_010.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_023_001.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_023_002.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_023_003.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_023_004.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_023_005.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_023_006.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_023_007.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_023_008.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_023_009.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_023_010.zip"
]
'''
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_024_001.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_024_002.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_024_003.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_024_004.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_024_005.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_024_006.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_024_007.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_024_008.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_024_009.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_024_010.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_024_011.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_024_012.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_024_013.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_024_014.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_024_015.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_024_016.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_024_017.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_024_018.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_024_019.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_026_001.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_026_002.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_026_003.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_026_004.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_026_005.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_026_006.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_026_007.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_026_008.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_026_009.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_026_011.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_026_012.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_026_013.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_026_014.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_026_015.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_026_016.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_026_017.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_026_018.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_026_019.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_026_020.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_027_001.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_027_003.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_027_004.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_027_005.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_027_006.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_027_007.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_027_008.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_027_009.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_027_010.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_028_001.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_028_002.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_028_003.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_028_004.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_028_005.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_028_006.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_028_008.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_028_009.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_029_001.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_029_002.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_029_003.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_029_004.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_029_005.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_030_001.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_030_002.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_030_003.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_030_004.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_030_005.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_030_007.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_030_008.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_030_009.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_030_010.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_031_001.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_031_003.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_031_004.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_031_006.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_031_007.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_031_008.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_031_009.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_031_010.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_032_001.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_032_002.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_032_003.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_032_004.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_032_005.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_032_007.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_032_008.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_032_009.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_033_001.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_033_002.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_033_004.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_033_005.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_033_007.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_033_008.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_033_009.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_033_010.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_034_001.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_034_002.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_034_003.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_034_005.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_035_001.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_035_002.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_035_003.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_035_004.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_035_005.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_035_006.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_035_007.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_035_008.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_035_009.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_035_010.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_036_001.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_036_002.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_036_003.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_036_005.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_036_006.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_036_007.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_036_008.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_036_010.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_037_001.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_037_002.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_037_003.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_037_004.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_037_005.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_037_006.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_037_007.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_037_008.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_037_009.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_037_010.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_038_002.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_038_004.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_038_005.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_038_006.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_038_007.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_038_009.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_038_010.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_039_002.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_039_003.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_039_004.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_039_005.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_039_006.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_039_007.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_039_008.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_039_009.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_039_010.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_041_001.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_041_002.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_041_003.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_041_004.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_041_005.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_041_006.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_041_007.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_041_008.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_041_009.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_041_010.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_042_001.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_042_002.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_042_003.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_042_004.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_042_005.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_043_002.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_043_003.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_043_004.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_043_005.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_043_006.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_043_007.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_043_008.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_043_009.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_043_010.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_044_001.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_044_002.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_044_003.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_044_004.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_044_005.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_044_006.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_044_007.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_044_008.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_044_009.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_044_010.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_045_001.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_045_004.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_045_005.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_045_006.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_045_008.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_045_010.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_046_001.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_046_002.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_046_003.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_046_004.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_046_005.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_046_006.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_046_007.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_046_008.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_047_001.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_047_002.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_047_003.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_047_004.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_047_005.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_047_006.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_047_007.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_047_008.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_047_009.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_048_001.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_048_002.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_048_003.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_048_004.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_048_005.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_048_006.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_048_007.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_048_008.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_048_009.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_048_010.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_050_001.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_050_002.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_050_003.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_050_004.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_050_005.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_051_001.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_051_002.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_051_003.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_051_004.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_051_005.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_052_001.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_052_002.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_052_003.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_052_004.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_052_005.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_052_006.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_052_007.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_052_008.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_052_009.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_052_010.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_053_001.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_053_002.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_053_003.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_053_004.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_053_005.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_053_006.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_053_007.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_053_008.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_053_009.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_053_010.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_053_012.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_053_013.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_053_014.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_053_016.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_053_017.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_053_018.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_053_019.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_053_020.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_054_001.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_054_002.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_054_003.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_054_004.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_054_005.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_054_006.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_054_007.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_054_008.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_054_009.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_054_010.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_055_001.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_055_002.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_055_003.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_055_004.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_055_005.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_055_006.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_055_007.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_055_008.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_055_009.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_055_010.zip",
]
'''
for url in urls_to_download:
    download(url)

print("[HYPERSIM: DATASET_DOWNLOAD_IMAGES] Finished.")

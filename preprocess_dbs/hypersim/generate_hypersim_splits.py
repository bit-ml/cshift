import os
import sys
import shutil
from cv2 import split
import pandas as pd

# main dataset path
DATASET_PATH = r'/data/multi-domain-graph/datasets/hypersim/data'

SPLITS_CSV_PATH = r'metadata_images_split_scene_v1_selection.csv'


def get_task_split_paths(dataset_path, splits_csv_path, split_name, folder_str,
                         task_str, ext):

    initial_split_name = 'train'
    df = pd.read_csv(splits_csv_path)
    df = df[df['included_in_public_release'] == True]
    df = df[df['split_partition_name'] == initial_split_name]

    train_index = -1
    if split_name.find('train') != -1:
        train_index = int(split_name[-1])
        split_name = 'train'
    paths = []
    scenes = df['scene_name'].unique()
    scenes = scenes[0:150]
    camera_index = -1
    sel_scenes = []
    sel_cameras = []
    sel_frames = []
    for scene in scenes:
        df_scene = df[df['scene_name'] == scene]
        cameras = df_scene['camera_name'].unique()

        if split_name == 'train':
            if (len(cameras) > 2):
                cameras = cameras[0:-1]
        else:
            if (len(cameras) > 2):
                cameras = cameras[-1:]
            else:
                cameras = []

        for camera in cameras:
            camera_index += 1
            if split_name == 'train' and train_index == 1 and camera_index % 2 == 1:
                continue
            if split_name == 'train' and train_index == 2 and camera_index % 2 == 0:
                continue
            if split_name == 'test' and camera_index % 4 == 0:
                continue
            if split_name == 'valid' and camera_index % 4 > 0:
                continue
            df_camera = df_scene[df_scene['camera_name'] == camera]
            frames = df_camera['frame_id'].unique()
            for frame in frames:
                path = '%s/%s/images/scene_%s_%s/frame.%04d.%s.%s' % (
                    dataset_path, scene, camera, folder_str, int(frame),
                    task_str, ext)
                paths.append(path)

                sel_scenes.append(scene)
                sel_cameras.append(camera)
                sel_frames.append(frame)
    return paths, sel_scenes, sel_cameras, sel_frames


def store_split(split_name, csv_file):

    _, scenes, cameras, frames = get_task_split_paths(DATASET_PATH,
                                                      SPLITS_CSV_PATH,
                                                      split_name,
                                                      'final_preview',
                                                      'tonemap', 'jpg')
    for i in range(len(scenes)):
        csv_file.write('%s,%s,%s,%s,\n' %
                       (split_name, scenes[i], cameras[i], frames[i]))


if __name__ == "__main__":
    train1_paths = get_task_split_paths(DATASET_PATH, SPLITS_CSV_PATH,
                                        'train1', 'final_preview', 'tonemap',
                                        'jpg')
    csv_file = open('./cshift_hypersim_splits.csv', 'w')
    csv_file.write('split,scene_name,camera_name,frame_id,\n')

    store_split('train1', csv_file)
    store_split('train2', csv_file)
    store_split('valid', csv_file)
    store_split('test', csv_file)

    csv_file.close()
import numpy as np
import torch
import yaml
import os
from glob import glob
import json
import hashlib
import shutil
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
import subprocess
import random
import copy
import multiprocessing
import pynvml
from utils.dataset import check_data, load_splits, save_splits, ObjectNeRF, Scene, Camera, ImageIO
from utils.prepare_data_utils import shared_argparser, default_output_root, initialize_lock, release_gpu
from utils.visualize_utils import SceneVisualizer
from collections import defaultdict
import itertools
from external import blenderproc_utils
from models.base import Front3DDataset
import networkx as nx
import mmcv
import cv2

def _hashing_split(ids, split):
    # generate weight by hashing id
    weight = []
    for i in ids:
        hash = np.frombuffer(hashlib.md5(i.encode('utf-8')).digest(), np.uint32)
        rng = np.random.RandomState(hash)
        weight.append(rng.random())

    # sort ids by weight
    ids = [i for _, i in sorted(zip(weight, ids))]

    # split ids
    split_ratios = list(split.values())
    split_index = [0] + np.round(np.cumsum(split_ratios) * len(ids)).astype(int).tolist()
    split_ids = {k: ids[split_index[i]:split_index[i+1]] for i, k in enumerate(split.keys())}
    
    return split_ids


def split_3d_front_scenes(args):
    # load config
    scene_data_config_dir = os.path.join(args.config_dir, 'scene_data.yaml')
    with open(scene_data_config_dir, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        
    # scan output folders
    output_root = default_output_root(args.output_dir, 'scene', args.debug)
    print(f"Scanning output folders in {output_root}...")
    front_scenes = glob(os.path.join(output_root, "*", "data.pkl"))
    
    # split scenes
    print(f"Splitting {len(front_scenes)} scenes...")
    scene_ids = [os.path.basename(os.path.dirname(scene)) for scene in front_scenes]
    split_scene_ids = _hashing_split(scene_ids, config['split'])

    # save scene_id splits to json
    for split_name, split_scene_id in split_scene_ids.items():
        print(f"Saving split {split_name} with {len(split_scene_id)} scenes...")
        with open(os.path.join(output_root, f'{split_name}.json'), 'w') as f:
            json.dump(split_scene_id, f, indent=4)


def split_3d_front_objects(args):
    # get data directory
    data_dirs = check_data(args.data_dir)
    
    # load config
    scene_data_config_dir = os.path.join(args.config_dir, 'object_data.yaml')
    with open(scene_data_config_dir, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    # load 3D-FUTURE objects info
    future_model_info_dir = os.path.join(data_dirs['future_dir'], 'model_info.json')
    with open(future_model_info_dir, 'r') as f:
        future_model_info = json.load(f)
    future_model_info = {model['model_id']: model for model in future_model_info}
    
    # scan 3d future folders
    print(f"Scanning 3d future folders in {data_dirs['future_dir']}...")
    front_objects = glob(os.path.join(data_dirs['future_dir'], '*/'))
    object_ids = [os.path.basename(os.path.normpath(dir)) for dir in front_objects]
    
    # map category
    obj_categories = defaultdict(list)
    for object_id in object_ids:
        obj_categories[str(future_model_info[object_id]['category'])].append(object_id)
    
    # split objects
    print(f"Splitting {len(front_objects)} objects...")
    splits = defaultdict(list)
    for category, object_ids in obj_categories.items():
        category_splits = _hashing_split(object_ids, config['split'])
        for split_name, split_object_ids in category_splits.items():
            splits[split_name] += split_object_ids

    # save object_id splits to json
    output_root = default_output_root(args.output_dir, 'object', args.debug)
    save_splits(output_root, splits)


def _filter_object_splits(args, data_folder):
    # load splits
    print('Loading object ids...')
    output_root = default_output_root(args.output_dir, data_folder, args.debug)
    splits = load_splits(output_root)

    # scan output folders
    print(f"Scanning output folders in {output_root}...")
    front_objects = glob(os.path.join(output_root, "*", "data.pkl"))

    # filter categories
    print(f"Filtering categories of {len(front_objects)} objects...")
    valid_front_objects = {}
    for front_object in tqdm(front_objects, desc='Filtering objects'):
        objnerf = ObjectNeRF.from_dir(os.path.dirname(front_object))
        if objnerf['category_id'] is not None:
            valid_front_objects[os.path.basename(os.path.dirname(front_object))] = objnerf['category']
    print(f"Found {len(valid_front_objects)} objects with valid categories...")

    # filter object ids
    for split_name, split_object_ids in splits.items():
        split_object_ids = [[i, valid_front_objects[i]] for i in split_object_ids if i in valid_front_objects]
        splits[split_name] = split_object_ids

    # overwrite splits
    save_splits(output_root, splits)


def filter_3d_front_object_splits(args):
    _filter_object_splits(args, 'object')


def filter_object_crop_splits(args):
    _filter_object_splits(args, 'object_crop')


def _cli_args_str(args):
    args_dict = {}
    bool_true_args = []
    for k, v in vars(args).items():
        if k in args.additional_args:
            continue
        if isinstance(v, bool):
            if v:
                bool_true_args.append(k)
        elif v is not None:
            args_dict[k] = v
    cli_args = ' '.join(['--' + k + ' ' + str(v) for k, v in args_dict.items()])
    cli_args += ' ' + ' '.join(['--' + k for k in bool_true_args])
    return cli_args


def _run_render_command(args):
    output_root = default_output_root(args.output_dir, args.job.split('_')[-1][:-1], args.debug)

    # use one gpu per process
    if args.processes > 0 and args.single_gpu:
        worker_id = multiprocessing.current_process()._identity[0] - 1
        if args.gpu_ids == 'all':
            pynvml.nvmlInit()
            num_gpus = pynvml.nvmlDeviceGetCount()
            pynvml.nvmlShutdown()
            args.gpu_ids = str(worker_id % num_gpus)
        else:
            gpu_ids = [int(gpu_id) for gpu_id in args.gpu_ids.split(',')]
            args.gpu_ids = str(gpu_ids[worker_id % len(gpu_ids)])
    
    # construct command
    cmd = f"blenderproc run utils/{args.job[:-1]}.py {_cli_args_str(args)}"
    
    output_id_dir = os.path.join(output_root, args.id)
    tqdm.write(f"Running command: {cmd}")
    args.retry = 0 if args.retry < 0 else args.retry
    for i in range(args.retry + 1):
        if i > 0:
            tqdm.write(f"Retry ({i}/{args.retry}) command: {cmd}")
        try:
            if args.debug:
                subprocess.run(cmd, shell=True, check=True)
            else:
                output_log_dir = os.path.join(output_id_dir, 'log.txt')
                os.makedirs(output_id_dir, exist_ok=True)
                with open(output_log_dir, 'a') as f:
                    subprocess.run(cmd, shell=True, stdout=f, stderr=f, check=True)
            return True
        except subprocess.CalledProcessError as e:
            tqdm.write(f"Command failed: {cmd}, this may be due to unknown GPU memory issue, return code: {e.returncode}")
        finally:
            release_gpu(output_root, args.gpu_ids, args.id)

    tqdm.write(f"Giving up command: {cmd}")
    return False


def _render_3d_front(args, task):
    # make output dir
    output_folder = 'scene' if task != 'object' else 'object'
    output_root = default_output_root(args.output_dir, output_folder, args.debug)

    # get data directory
    if task == 'scene':
        # get scene ids for rendering
        if args.id is not None:
            ids = set([args.id])
        else:
            data_dirs = check_data(args.data_dir)
            print('Scanning scene ids...')
            dirs = glob(os.path.join(data_dirs['front_scene_dir'], '*.json'))
            ids = set(os.path.basename(os.path.normpath(dir)).split('.')[0] for dir in dirs)
        
        # skip previously skipped ids if debug is not set
        skipped_scene_dir = os.path.join(output_root, 'skipped_scenes.txt')
        if not args.debug and os.path.exists(skipped_scene_dir):
            skipped_scenes = set()
            with open(skipped_scene_dir, 'r') as f:
                for line in f:
                    skipped_scenes.add(line.strip())
            print(f"Previously skipped {len(skipped_scenes)} invalid scenes.")
            ids = ids - skipped_scenes
        
        ids = list(ids)
        ids.sort()

    elif task == 'object':
        # load config
        object_data_config_dir = os.path.join(args.config_dir, 'object_data.yaml')
        with open(object_data_config_dir, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            
        # load splits
        print('Loading objct ids...')
        splits = load_splits(output_root, config['split'])
        
        # get object id and split pairs for rendering
        if args.id is not None:
            for split, ids in splits.items():
                if args.id in ids:
                    break
            ids = [(args.id, split)]
        else:
            ids = [(i, split) for split, ids in splits.items() for i in ids]
            ids = sorted(ids, key=lambda x: x[0])

    elif task == 'full_image':
        if args.id is not None:
            ids = set([args.id])
        else:
            print('Scanning scene ids...')
            dirs = glob(os.path.join(output_root, '*', '*.pkl'))
            ids = set(os.path.dirname(os.path.normpath(dir)).split('/')[-1] for dir in dirs)

        ids = list(ids)
        ids.sort()
    else:
        raise ValueError(f"scene_or_object must be 'scene' or 'object', got {task}")

    print(f"Found {len(ids)} {task} ids.")

    # copy config files
    if task != 'full_image':
        for other_config in (f"{task}_data", 'object_category2future', 'object_categories'):
            shutil.copy(os.path.join(args.config_dir, f"{other_config}.yaml"), output_root)
    else:
        for other_config in ('object_category2future', 'object_categories'):
            shutil.copy(os.path.join('data/object', f"{other_config}.yaml"), output_root)

    # scan existing outputs if overwrite not set
    if not args.overwrite and task != 'full_image':
        print(f"Scanning existing {task} outputs...")
        existing_output_dirs = glob(os.path.join(output_root, '*', 'data.pkl'))
        existing_output_ids = set(os.path.basename(os.path.normpath(os.path.dirname(dir))) for dir in existing_output_dirs)
    else:
        existing_output_ids = []

    # construct args for each render
    random.seed(args.seed)
    args_list = []
    print('Constructing args...')
    for i in ids:
        if task == 'object':
            i, args.split = i
        # skip if overwrite not set and output file exists
        if not args.overwrite and i in existing_output_ids:
            continue
        # construct args
        args.seed = random.randint(0, 2**32 - 1)
        args.id = i
        args_list.append(copy.deepcopy(args))
    print(f"Skipped {len(ids) - len(args_list)} existing {task}s")
    print(f"Total {len(args_list)} {task}s to render")
    
    # start rendering
    message = f"Rendering 3D-FRONT {task}s"
    initialize_lock(output_root)
    if args.processes == 0:
        # single process
        success = [_run_render_command(a) for a in tqdm(args_list, desc=message)]
    else:
        # render in parallel
        success = process_map(_run_render_command, args_list, desc=message, max_workers=args.processes, chunksize=1)
    
    # print summary
    if all(success):
        print(f"Successfully rendered all {task}s")
    else:
        # print and save failed ids
        print(f"Rendered {sum(success)} {task}s")
        print(f"The below {len(success) - sum(success)} {task}s are failed:")
        failed_dir = os.path.join(output_root, f"failed_{task}.txt")
        sample = 0
        with open(failed_dir, 'w') as f:
            for args, s in zip(args_list, success):
                if not s:
                    if sample < 10:
                        print(args.id)
                        sample += 1
                        if sample == 10:
                            print('...')
                    f.write(args.id + '\n')
        print(f"Failed {task} are saved to {failed_dir}")

    # deduplicate and print skipped scenes
    if task == 'scene' and os.path.exists(skipped_scene_dir):
        skipped_scenes = set()
        with open(skipped_scene_dir, 'r') as f:
            for line in f:
                skipped_scenes.add(line.strip())
        skipped_scenes = list(skipped_scenes)
        skipped_scenes.sort()
        print(f"The below {len(skipped_scenes)} scenes are invalid:")
        sample = 0
        with open(skipped_scene_dir, 'w') as f:
            for s in skipped_scenes:
                if sample < 10:
                    print(s)
                    sample += 1
                    if sample == 10:
                        print('...')
                f.write(s + '\n')
        print(f"Invalid scenes are saved to {skipped_scene_dir}")


def render_3d_front_scenes(args):
    _render_3d_front(args, 'scene')


def render_3d_front_objects(args):
    _render_3d_front(args, 'object')


def render_full_image_scenes(args):
    _render_3d_front(args, 'full_image')


def _crop_object_images_from_single_scene(args):
    output_root = default_output_root(args.output_dir, 'object_crop', args.debug)

    # load scene
    scene_dir = os.path.join(args.data_dir, args.id)
    tqdm.write(f"Loading scene from {scene_dir}")
    scene = Scene.from_dir(scene_dir)
    if args.min_object_area:
        scene.remove_small_object2d(args.min_object_area)

    # crop images for each object
    scene.crop_object_images()
    scene.aggregate_object_images()
    obj_scene_ids = []

    # skip if no object
    if not scene['object']:
        return obj_scene_ids

    for obj_id, objnerf in scene['object'].items():
        # skip if no camera
        if not objnerf['camera']:
            continue

        # save data
        obj_scene_id = f"{objnerf['jid']}.{scene['uid']}"
        obj_scene_ids.append(obj_scene_id)
        output_object_dir = os.path.join(output_root, obj_scene_id)
        # first remove output scene directory to update the data
        shutil.rmtree(output_object_dir, ignore_errors=True)
        # save data
        objnerf.save(output_object_dir)

    return obj_scene_ids


def crop_object_images_from_scene(args):
    # debug mode
    if args.debug:
        args.data_dir += '-debug'
    output_root = default_output_root(args.output_dir, 'object_crop', args.debug)

    # copy config files
    for other_config in ('object_category2future', 'object_categories'):
        shutil.copy(os.path.join(args.data_dir, f"{other_config}.yaml"), output_root)

    # crop single scene if specified
    if args.id is not None:
        _crop_object_images_from_single_scene(args)

    # load splits
    print('Loading object ids...')
    splits = load_splits(args.data_dir)

    obj_splits = {}
    for split_name, scene_ids in splits.items():
        # construct args for each scene
        args_list = []
        for scene_id in scene_ids:
            args.id = scene_id
            args_list.append(copy.deepcopy(args))

        message = f"Cropping images for {split_name} split"
        if args.processes == 0:
            obj_scene_ids = [
                _crop_object_images_from_single_scene(a) for a in tqdm(args_list, desc=message)]
        else:
            obj_scene_ids = process_map(
                _crop_object_images_from_single_scene, args_list, desc=message, max_workers=args.processes)
        obj_splits[split_name] = sorted(itertools.chain.from_iterable(obj_scene_ids))

    # save split
    save_splits(output_root, obj_splits)


def calculate_normalization_params(args):
    # load splits
    print('Loading scene ids...')
    output_root = default_output_root(args.output_dir, 'scene', args.debug)
    splits = load_splits(output_root)

    # calculate normalization params for train split
    total_size = defaultdict(lambda: 0.)
    num_objects = defaultdict(lambda: 0)
    for scene_id in tqdm(splits['train'], desc='Calculating normalization params'):
        scene_dir = os.path.join(output_root, scene_id)
        scene = Scene.from_dir(scene_dir)

        # calculate normalization params
        for obj_3d in scene['object'].values():
            total_size[obj_3d['category']] += obj_3d['size']
            num_objects[obj_3d['category']] += 1

    # save normalization params
    mean_size = {k: (total_size[k] / num_objects[k]).tolist() for k in total_size}
    with open(os.path.join(output_root, 'object_mean_size.json'), 'w') as f:
        json.dump(mean_size, f, indent=4)


def link_depth_dataset(args):
    # load splits
    print('Loading scene ids...')
    scene_root = default_output_root(args.data_dir, 'scene', args.debug)
    splits = load_splits(scene_root)
    splits['all'] = sum(list(splits.values()), [])  # add 'all' split for generating depth dataset
    
    # output directory
    output_root = default_output_root(args.output_dir, 'depth', args.debug)
    shutil.rmtree(output_root, ignore_errors=True)

    for split_name, scene_ids in splits.items():
        # create directories
        split_dir = os.path.join(output_root, split_name)
        os.makedirs(split_dir, exist_ok=True)
        depth_dir = os.path.join(split_dir, 'depth')
        os.makedirs(depth_dir, exist_ok=True)
        rgb_dir = os.path.join(split_dir, 'rgb')
        os.makedirs(rgb_dir, exist_ok=True)

        # create symbolic links
        for scene_id in tqdm(scene_ids, desc=f"Linking depth dataset for {split_name} split"):
            scene_dir = os.path.join(scene_root, scene_id)
            scene = Scene.from_dir(scene_dir)

            for cam_id, camera in scene['camera'].items():
                image = dict(camera['image'])
                if 'depth' in image:
                    os.link(image['depth'], os.path.join(depth_dir, f"{scene_id}.{cam_id:04d}.png"))
                os.link(image['color'], os.path.join(rgb_dir, f"{scene_id}.{cam_id:04d}.png"))


def link_depth_output(args):
    # scan depth outputs
    print('Scanning depth outputs...')
    depth_dirs = glob(os.path.join(args.input_dir, '*.png'))
    scene_depth_dirs = defaultdict(dict)
    for depth_dir in depth_dirs:
        depth_name = os.path.splitext(os.path.basename(depth_dir))[0]
        scene_id, cam_id = depth_name.split('.')
        scene_depth_dirs[scene_id][int(cam_id)] = depth_dir

    # load splits
    print('Loading scene ids...')

    data_dir = "data_debug"
    output_dir = "data_debug"

    scene_root = default_output_root(data_dir, 'scene', args.debug)
    splits = load_splits(scene_root)

    output_root = default_output_root(output_dir, 'scene-monocular_depth', args.debug)
    save_splits(output_root, splits)

    # copy config files
    for other_config in ('object_category2future.yaml', 'object_categories.yaml', 'object_mean_size.json'):
        if os.path.exists(os.path.join(scene_root, other_config)):
            shutil.copy(os.path.join(scene_root, other_config), output_root)

    # link depth outputs
    #scene_ids = sum(list(splits.values()), [])
    scene_ids = '51508cc8-618a-4537-b753-53a381c1af0d'
    for scene_id in tqdm(scene_ids, desc='Linking depth outputs'):
        # create output directory
        output_scene_dir = os.path.join(output_root, scene_id)
        shutil.rmtree(output_scene_dir, ignore_errors=True)
        os.makedirs(output_scene_dir, exist_ok=True)

        # link depth image
        assert scene_id in scene_depth_dirs, f"Depth output for scene {scene_id} not found."
        for cam_id, depth_dir in scene_depth_dirs[scene_id].items():
            os.link(depth_dir, os.path.join(output_scene_dir, f"{cam_id:04d}-depth.png"))

        # link other files from original dataset
        input_scene_dir = os.path.join(scene_root, scene_id)
        for file_dir in glob(os.path.join(input_scene_dir, '*')):
            if not os.path.splitext(os.path.basename(file_dir))[0].endswith('depth'):
                os.link(file_dir, os.path.join(output_scene_dir, os.path.basename(file_dir)))# fintune之后的depth


def generate_coco_dataset(args):
    # load splits
    print('Loading scene ids...')
    scene_root = default_output_root(args.data_dir, 'scene', args.debug)
    splits = load_splits(scene_root)
    splits['all'] = sum(list(splits.values()), [])

    # output directory
    output_root = default_output_root(args.output_dir, 'detection', args.debug)
    shutil.rmtree(output_root, ignore_errors=True)

    for split_name, scene_ids in splits.items():
        # create directories
        split_dir = os.path.join(output_root, split_name)
        image_dir = os.path.join(split_dir, 'images')
        os.makedirs(image_dir, exist_ok=True)

        annotations = {"images": [], "annotations": []}
        image_id = 0
        obj_id = 0
        for scene_id in tqdm(scene_ids, desc=f"Generating coco dataset for {split_name} split"):
            scene_dir = os.path.join(scene_root, scene_id)
            scene = Scene.from_dir(scene_dir)

            for cam_id, camera in scene['camera'].items():
                # create symbolic links
                image = dict(camera['image'])
                image_file_name = f"{scene_id}.{cam_id:04d}.png"
                os.link(image['color'], os.path.join(image_dir, image_file_name))

                # generate annotations for images
                annotations["images"].append({
                    "id": image_id,
                    "width": camera.get('width', scene['camera'][cam_id]['image']['color'].shape[1]),
                    "height": camera.get('height', scene['camera'][cam_id]['image']['color'].shape[0]),
                    "file_name": os.path.join('images', image_file_name),
                })

                # generate annotations for objects
                if 'object' in camera:
                    for obj in camera['object'].values():
                        annotations['annotations'].append({
                            "id": obj_id,
                            "image_id": image_id,
                            "category_id": obj['category_id'],
                            "segmentation": blenderproc_utils.binary_mask_to_polygon(obj['segmentation']),
                            "area": obj['area'],
                            "bbox": obj['bdb2d'].tolist(),
                            "iscrowd": 0,
                        })
                        obj_id += 1

                image_id += 1

        # create category annotations
        if scene.object_categories:
            object_categories = scene.object_categories
        else:
            coco_category_config = os.path.join(args.config_dir, 'object_coco_categories.yaml')
            with open(coco_category_config) as f:
                object_categories = yaml.load(f, Loader=yaml.FullLoader)
        annotations["categories"] = [{
            "id": i,
            "name": c,
            "supercategory": "coco_annotations",
        } for i, c in enumerate(object_categories)]

        # save annotations
        with open(os.path.join(split_dir, 'coco_annotations.json'), 'w') as f:
            json.dump(annotations, f, indent=4)


def generate_scene_dataset_from_detection_result(args):
    import pycocotools.mask as mask_util

    input_dir = os.path.join("gseg/outputs",args.scene_id,"mask_rcnn.json")
    # read detection results
    print('Reading detection results...')
    with open(input_dir) as f:
        segm_results = json.load(f)

    # load image ids
    print('Loading image ids...')
    # detection_root = default_output_root(args.data_dir, 'detection', args.debug)
    # with open(os.path.join(detection_root, 'all/coco_annotations.json')) as f:
    #     coco_annotations = json.load(f) 
    scene_cam_idx = defaultdict(dict)

    scene_id = args.scene_id
    for i in range(10):
        scene_cam_idx[scene_id][i] = i
    # for image in coco_annotations['images']:
    #     scene_id, cam_id = os.path.basename(image['file_name']).split('.')[:2]
    #     scene_cam_idx[scene_id][int(cam_id)] = image['id']

    # load splits
    # generate scene undrstanding data:
    print('Loading scene ids...')

    data_dir = "data_debug"
    output_dir = "data_debug"

    scene_root = default_output_root(data_dir, 'scene-monocular_depth', args.debug)
    splits = load_splits(scene_root)

    output_root = default_output_root(output_dir, 'scene-detector', args.debug)
    save_splits(output_root, splits)

    # copy config files
    for other_config in ('object_category2future', 'object_categories'):
        if os.path.exists(os.path.join(args.config_dir, f"{other_config}.yaml")):
            shutil.copy(os.path.join(args.config_dir, f"{other_config}.yaml"), output_root)

    # load coco category mapping
    if args.nyuv2:
        coco_category_config = os.path.join(args.config_dir, 'object_coco_categories.yaml')
        with open(coco_category_config) as f:
            coco_categories = yaml.load(f, Loader=yaml.FullLoader)
        category_mapping_config = os.path.join(args.config_dir, 'object_category2coco.yaml')
        with open(category_mapping_config) as f:
            category2coco = yaml.load(f, Loader=yaml.FullLoader)
        coco2category = {v: k for k, v in category2coco.items()}
        coco_id2category = {i: coco2category[c] for i, c in enumerate(coco_categories) if c in coco2category}

    #  covert bbox
    def convert(bbox):
        w = np.abs(bbox[2] - bbox[0])
        h = np.abs(bbox[3] - bbox[1])
        return (bbox[0],bbox[1], w, h)

    # generate scene dataset from detection results
    scene_ids = sum(list(splits.values()), [])
    for scene_id in tqdm(scene_ids, desc='Generating dataset'):
        # create output directory
        output_scene_dir = os.path.join(output_root, scene_id)
        shutil.rmtree(output_scene_dir, ignore_errors=True)
        os.makedirs(output_scene_dir, exist_ok=True)

        # link images
        input_scene_dir = os.path.join(scene_root, scene_id)
        input_scene = Scene.from_dir(input_scene_dir)

        pic_dir = glob(os.path.join(input_scene_dir, '*.png'))
        for pic in pic_dir:
            catagory = ((pic.split('.')[0]).split('/')[-1]).split('-')[-1]
            pic_name = pic.split('/')[-1]
            if catagory == 'instance_segmap':
                continue
            os.link(pic, os.path.join(output_scene_dir, pic_name))

        # split detection results
        image_segm_results = {}
        for segm_result in segm_results:
            idx = range(0, 20)
            result = dict(zip(idx,segm_result[1:]))
            image_segm_results[segm_result[0]['image_id']] = result 

        # generate data.pkl from detection results

        category = input_scene['camera'][0]['object'][0]['category']
        data = {'uid': scene_id, 'camera': {}}
        for cam_id, image_idx in scene_cam_idx[scene_id].items():
            # generate object annotations from detection results
            input_camera = input_scene['camera'][cam_id]

            image_segm_result = image_segm_results[image_idx]
            obj2d_annotations = {}
            for obj_id, segm_result in image_segm_result.items():
                if args.nyuv2 and segm_result['category'] not in coco_id2category:
                    continue
                segmentation = torch.Tensor(segm_result['segmentation']).bool()
                obj2d_annotations[obj_id] = {
                    'category': category,
                    'bdb2d': np.array(convert(segm_result['bdb2d'])).round().astype(int),
                    'score': segm_result['score'],
                    'segmentation': segmentation,
                    'area': blenderproc_utils.calc_binary_mask_area(segmentation),
                }
                if args.nyuv2:
                    obj2d_annotations[obj_id]['category'] = coco_id2category[segm_result['category']]


            data['camera'][cam_id] = {
                'K': input_camera.get(
                    'K',
                    np.array([  # default K matrix for NYUv2
                        [5.1885790117450188e+02, 0, 3.2558244941119034e+02],
                        [0, 5.1946961112127485e+02, 2.5373616633400465e+02],
                        [0, 0, 1],
                    ],dtype=np.float32)
                ),
                'height': input_camera.get('height', input_camera['image']['color'].shape[0]),
                'width': input_camera.get('width', input_camera['image']['color'].shape[1]),
                'object': obj2d_annotations,
            }
        
        # save scene data
        scene = Scene(data)
        scene.save(output_scene_dir)


def generate_scene_understanding_dataset(args):
    random.seed(args.seed)

    scene_data_config_dir = os.path.join(args.config_dir, 'scene_understanding_data.yaml')
    with open(scene_data_config_dir, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    output_root = default_output_root(args.output_dir, 'scene-understanding', args.debug)

    # copy config files
    for other_config in ('object_category2future', 'object_categories'):
        shutil.copy(os.path.join(config['dataset']['dir'], f"{other_config}.yaml"), output_root)

    # filter scene ids
    dataset = Front3DDataset(config['dataset'], 'test')

    scene_ids = []
    for i in tqdm(range(len(dataset)), desc='Generating dataset'):
        input_scene, gt_scene = dataset.get_scene(i)
        scene_id = input_scene['uid'] 

        # create output directory
        output_scene_dir = os.path.join(output_root, scene_id)
        shutil.rmtree(output_scene_dir, ignore_errors=True)
        os.makedirs(output_scene_dir, exist_ok=True)

        # select cameras for input and evaluation
        ref_camera = config['known_camera']['ref_camera']
        skip_scene = True
        for min_overlap in range(config['dataset']['min_overlap_object'], 0, -1):
            for _ in range(100):
                ref_subscene = gt_scene.random_subscene(ref_camera, min_overlap_object=config['dataset']['min_overlap_object'])
                novel_subscene = gt_scene.subscene(set(gt_scene['camera'].keys()) - set(ref_subscene['camera'].keys()))
                # make sure that novel views contain at least one object in the input views
                input_objects = set(ref_subscene['object'].keys())
                if all(len(set(cam['object'].keys()) & input_objects) >= min_overlap for cam in novel_subscene['camera'].values()):
                    skip_scene = False
                    break
            if skip_scene is False:
                break
        if skip_scene:
            tqdm.write(f"Failed to select novel views for scene {scene_id}")
            continue

        # select cameras with camera numbers from min_camera to ref_camera
        chosen_cam_id = {ref_camera: list(ref_subscene['camera'].keys()), 'ref_camera': ref_camera}
        for num_cam in range(ref_camera - 1, config['known_camera']['min_camera'] - 1, -1):
            input_subscene = ref_subscene.random_subscene(num_cam, min_overlap_object=config['dataset']['min_overlap_object'])
            chosen_cam_id[num_cam] = list(input_subscene['camera'].keys())

        # select cameras with camera numbers from ref_camera to max_camera
        camera_overlap_graph = gt_scene.camera_overlap_graph
        input_cam_id = set(chosen_cam_id[ref_camera])
        for num_cam in range(ref_camera + 1, config['known_camera']['max_camera'] + 1):
            neighbors = list(nx.node_boundary(camera_overlap_graph, input_cam_id))
            if neighbors:
                neighbor_to_add = random.choice(neighbors)
                input_cam_id.add(neighbor_to_add)
                chosen_cam_id[num_cam] = list(input_cam_id)
        chosen_cam_id['novel'] = list(set(gt_scene['camera'].keys()) - input_cam_id)

        # save selected cameras for input and evaluation
        input_scene['chosen_cam_id'] = chosen_cam_id

        # mask out background in input views using estimated mask
        for cam_id in input_scene['camera'].keys():
            gt_cam, in_cam = gt_scene['camera'][cam_id], input_scene['camera'][cam_id]
            masked_color = gt_cam['image']['color'].copy()
            background = np.ones(masked_color.shape[:2], dtype=bool)
            for obj_id, obj2d in gt_cam['object'].items():
                if obj_id in input_objects:
                    background[obj2d['segmentation']] = False
            masked_color[background] = 1.
            in_cam['image']['masked_color'] = masked_color

        # save scene data
        input_scene.save(output_scene_dir)

        # save chosen_cam_id as json
        with open(os.path.join(output_scene_dir, 'chosen_cam_id.json'), 'w') as f:
            json.dump(chosen_cam_id, f)

        scene_ids.append(scene_id)

    save_splits(output_root, {'test': scene_ids})


def generate_scene_from_colmap_pose(args):
    print('Loading scene ids...')
    scene_root = default_output_root(args.data_dir, 'scene', args.debug)
    colmap_root = default_output_root(args.data_dir, 'scene-understanding-blender', args.debug)
    splits = load_splits('configs', ['colmap_valid_scenes'])
    output_root = default_output_root(args.output_dir, 'scene-colmap', args.debug)
    save_splits(output_root, {'test': splits['colmap_valid_scenes']})

    for scene_id in tqdm(splits['colmap_valid_scenes'], desc='Generating camera poses'):
        colmap_dir = os.path.join(colmap_root, scene_id, 'json', 'transform_info.json')
        gt_scene_dir = os.path.join(scene_root, scene_id)
        output_scene_dir = os.path.join(output_root, scene_id)
        gt_scene = Scene.from_dir(gt_scene_dir)

        # load camera poses for input estimated witl colmap
        with open(colmap_dir, 'r') as f:
            transform_info = json.load(f)
        input_scene = Scene({'camera': {}})
        for cam in transform_info.values():
            cam_id = int(os.path.splitext(cam['image_name'])[0].split('_')[-1])
            transform_mat = np.array(cam['transform_matrix'])
            transform_mat = np.concatenate([transform_mat, np.array([0, 0, 0, 1]).reshape(1, 4)], axis=0)
            input_scene['camera'][cam_id] = gt_scene['camera'][cam_id].copy()
            input_scene['camera'][cam_id]['cam2world_mat'] = transform_mat

        # set world coordinate frame to the camera with most objects
        gt_input_scene = gt_scene.subscene(input_scene['camera'].keys())
        origin_camera_id = max(gt_input_scene['camera'], key=lambda k: len(gt_input_scene['camera'][k]['object']))
        gt_scene.set_world_coordinate_frame(origin_camera_id)
        gt_input_scene.set_world_coordinate_frame(origin_camera_id)

        # generate transformation matrix from origin camera to gt origin camera
        gt_origin_cam = gt_scene['camera'][origin_camera_id]
        origin_cam = input_scene['camera'][origin_camera_id]
        transform_mat = gt_origin_cam['cam2world_mat'] @ np.linalg.inv(origin_cam['cam2world_mat'])

        # randomly select a camera to scale the scene
        scale_camera_id = random.choice(list(set(input_scene['camera'].keys()) - {origin_camera_id}))
        gt_scale_cam = gt_scene['camera'][scale_camera_id]
        scale_cam = input_scene['camera'][scale_camera_id]
        gt_scale = np.linalg.norm(gt_scale_cam['position'] - gt_origin_cam['position'])
        scale = np.linalg.norm(scale_cam['position'] - origin_cam['position'])
        scale_factor = gt_scale / scale

        # transform input camera poses to match gt scene
        for cam in input_scene['camera'].values():
            cam['cam2world_mat'] = transform_mat @ cam['cam2world_mat']
            cam['position'] = cam['position'] * scale_factor

        # visualize and save transformed scene
        shutil.rmtree(output_scene_dir, ignore_errors=True)
        os.makedirs(output_scene_dir, exist_ok=True)
        # gt_input_vis = SceneVisualizer(gt_input_scene)
        # input_vis = SceneVisualizer(input_scene)
        # input_vis.scene_vis(output_scene_dir, write_image=True, reference_scene_vis=gt_input_vis)
        colmap_scene = gt_scene.copy()
        colmap_scene['camera'].update(input_scene['camera'])
        gt_vis = SceneVisualizer(gt_scene)
        colmap_vis = SceneVisualizer(colmap_scene)
        colmap_vis.scene_vis(output_scene_dir, write_image=True, reference_scene_vis=gt_vis)
        colmap_scene.save(output_scene_dir)

def generate_depth_data(args):
    scene_id = args.scene_id
    scene_dir = os.path.join("data_debug/scene",scene_id)
    pic_dir = glob(os.path.join(scene_dir, '*.png'))  
    output_root = 'Monocular-Depth-Estimation-Toolbox-main/data'
    output_pic_dir = os.path.join(output_root, scene_id)
    shutil.rmtree(output_pic_dir , ignore_errors=True)
    os.makedirs(output_pic_dir, exist_ok=True)
    for pic in pic_dir:
        catagory = ((pic.split('.')[0]).split('/')[-1]).split('-')[-1]
        pic_name = pic.split('/')[-1]
        if catagory == 'depth':
            os.makedirs(os.path.join(output_pic_dir,'depth'), exist_ok=True)
            os.link(pic, os.path.join(output_pic_dir,'depth', pic_name))
        if catagory == 'color':
            os.makedirs(os.path.join(output_pic_dir,'rgb'), exist_ok=True)
            os.link(pic, os.path.join(output_pic_dir,'rgb', pic_name))

def main():
    parser = shared_argparser()
    parser.add_argument('--job', type=str, nargs='+', help='Job(s) to run.', default=[], required=True)
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing scene or object output.')
    parser.add_argument('--processes', type=int, default=0, help='Number of threads. Should be set regarding to the number of CPU cores and memory. 0 means no parallelization.')
    parser.add_argument('--retry', type=int, default=2, help='Times to retry if render failed.')
    parser.add_argument('--single_gpu', action='store_true', help='Use single GPU for each thread.')
    parser.add_argument('--input_dir', type=str, default=None, help='Input data directory.')
    parser.add_argument('--scene_id', type=str, default=None, help='Input data id.')
    parser.add_argument('--min_object_area', type=float, default=None, help='Minimum object area when generating object crops.')
    parser.add_argument('--nyuv2', action='store_true', help='Use coco categories on NYUv2 dataset.')
    args = parser.parse_args()
    args.additional_args = ['additional_args', 'job','scene_id','overwrite', 'processes', 'retry', 'single_gpu']
    
    # run jobs by names
    jobs = args.job
    for job in jobs:
        args.job = job
        print(f"Running job {job}...")
        globals()[job](copy.deepcopy(args))


if __name__ == "__main__":
    main()

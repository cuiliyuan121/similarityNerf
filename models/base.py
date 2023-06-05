from torch.utils.data import Dataset
import copy
import json
from glob import glob
import os
from utils.dataset import CategoryMapping, Scene, GetFailedError
from utils.torch_utils import MyLightningModule
import wandb
from tqdm import tqdm
import torch
import shutil
from utils.visualize_utils import image_float_to_uint8, SceneVisualizer, image_grid
import numpy as np
from torch import nn
import torchmetrics
import trimesh
from collections import defaultdict
from utils.transform import homotrans, bdb3d_corners, cam2uv, uv2cam, bbox_from_binary_mask, BDB3D_FACES, rotation_mat_dist
from utils.metrics import bdb3d_iou, seg_iou
from contextlib import suppress
import networkx as nx
from PIL import Image
import os
import tempfile


class Front3DDataset(Dataset):
    def __init__(self, config, split):
        self.config = config
        self.split = split if config['dir'] else 'predict'

        # default settings for null values
        self.config['min_object'] = self.config['min_object'] or 0
        self.config['min_overlap_object'] = self.config['min_overlap_object'] or 0
        self.config['min_camera'] = self.config['min_camera'] or 1

        # load split
        split_dir = os.path.join(config['input_dir'] or config['dir'], f'{split}.json')
        if config['ids']:
            # use specified ids
            if isinstance(config['ids'], str):
                with open(config['ids']) as f:
                    self.scene_ids = json.load(f)
            else:
                self.scene_ids = config['ids']
            print(f"Using the specified {len(self.scene_ids)} scenes: {self.scene_ids}")
            self.is_result = not os.path.exists(split_dir)
        else:
            if os.path.exists(split_dir):
                # load split from the dataset
                with open(split_dir) as f:
                    self.scene_ids = json.load(f)
                self.scene_ids.sort()
                print(f"Got {len(self)} scenes for {split} split")
                self.is_result = False
            else:
                # scan input data if needed
                input_scenes = glob(os.path.join(config['input_dir'], "*", "data.pkl"))
                self.scene_ids = [os.path.basename(os.path.dirname(scene)) for scene in input_scenes]
                print(f"Got {len(self)} scenes for input data")
                self.is_result = True

        # load category mapping
        CategoryMapping.load_category_mapping(self.config['dir'])

        # filter scene ids
        if not config['ids'] and not self.is_result:
            self.filter_invalid_scene()

    def __len__(self):
        return len(self.scene_ids)

    def set_world_coordinate_frame(self, input_scene, gt_scene):
        # if the origin camera id is specified, use it
        if 'origin_camera_id' in input_scene:
            gt_scene.set_world_coordinate_frame(input_scene['origin_camera_id'])
            return input_scene, gt_scene

        # otherwise, set the origin camera id to the camera with the largest number of objects
        has_object_gt = 'object' in next(iter(gt_scene['camera'].values()))
        has_object_input = 'object' in next(iter(input_scene['camera'].values()))
        if has_object_input or has_object_gt:
            # set world coordinate frame to the camera with the largest number of objects
            ref_scene = gt_scene if has_object_gt else input_scene
            if len(input_scene['camera']) != len(gt_scene['camera']):
                ref_scene = gt_scene.subscene(input_scene['camera'].keys())
            origin_camera_id = max(ref_scene['camera'], key=lambda k: len(ref_scene['camera'][k]['object']))
            input_scene.set_world_coordinate_frame(origin_camera_id)
            gt_scene.set_world_coordinate_frame(origin_camera_id)
        else:
            # set world coordinate frame to the camera in the middle
            origin_camera_id = gt_scene.set_world_coordinate_frame()
            input_scene.set_world_coordinate_frame(origin_camera_id)
        return input_scene, gt_scene

    def get_scene(self, idx):
        # load scene
        if self.config['dir'] is not None:
            gt_scene_dir = os.path.join(self.config['dir'], self.scene_ids[idx])
            gt_scene = Scene.from_dir(gt_scene_dir)
            gt_scene['split'] = self.split
            gt_scene.remove_small_object2d(self.config['min_object_area'])

        if self.config['input_dir'] is not None:
            # load input scene
            input_scene_dir = os.path.join(self.config['input_dir'], self.scene_ids[idx])
            input_scene = Scene.from_dir(input_scene_dir)
            input_scene['split'] = self.split
            # for cam_id, cam in cam_temp.items():
            #     input_scene['camera'][cam_id]['image']['depth'] = cam[]
            input_scene.remove_small_object2d(self.config['min_object_area'])
            if self.config['dir'] is None:
                gt_scene = input_scene.copy()

            if 'chosen_cam_id' in input_scene:
                if self.config['max_camera'] == 1:
                    input_cam_id = []
                    input_cam_id.append(input_scene['origin_camera_id'])
                elif self.config['max_camera'] == 10:
                    input_cam_id = [id_ for id_ in input_scene['camera'].keys()]
                else:
                    input_cam_id = input_scene['chosen_cam_id'][self.config['max_camera']]
                if self.config['dir'] is None:
                    gt_cam_id = input_cam_id
                elif self.config['full_novel_view']:
                    gt_cam_id = list(input_scene['camera'].keys())
                else:
                    gt_cam_id = input_cam_id + input_scene['chosen_cam_id']['novel']
                input_scene = input_scene.subscene(input_cam_id)
                gt_scene = gt_scene.subscene(gt_cam_id)
            elif self.is_result:
                gt_scene = gt_scene.subscene(input_scene['camera'].keys())

            if self.config['dir'] is None:
                return self.set_world_coordinate_frame(input_scene, gt_scene)

            if 'chosen_cam_id' in input_scene or self.is_result:
                return self.set_world_coordinate_frame(input_scene, gt_scene)

            # find valid camera ids with enough objects in both scenes
            valid_cam_ids = set(input_scene['camera'].keys()) & set(gt_scene['camera'].keys())
            input_scene = input_scene.subscene(valid_cam_ids)
            gt_scene = gt_scene.subscene(valid_cam_ids)

        # sample gt_scene according to camera_settings
        if self.config['cameras'] and len(self.config['cameras']) > idx:
            # use specified camera ids
            gt_scene = gt_scene.subscene(self.config['cameras'][idx][:self.config['max_camera']])
        else:
            # remove cameras with too small depth
            # gt_scene.proximity_check_camera(self.config.get('min_depth', None))
            if not gt_scene['camera']:
                return None, None

            # randomly sample cameras
            max_camera = self.config['max_camera'] or len(gt_scene['camera'])

            if self.config['max_object']:
                # if max_object is specified, sample gt_scene until the number of objects is less than max_object
                # but min_camera must be satisfied
                sample_scene = gt_scene
                num_camera = max_camera
                while True:
                    sample_scene = gt_scene.random_subscene(
                        num_camera, min_overlap_object=self.config['min_overlap_object'], min_object=self.config['min_object'])
                    num_camera = len(sample_scene['camera']) - 1
                    num_object = sum(len(camera['object']) for camera in sample_scene['camera'].values())
                    if num_object <= self.config['max_object'] or num_camera < (self.config['min_camera']):
                        break
                gt_scene = sample_scene
            else:
                # if max_object is not specified, sample gt_scene with given max_camera, min_overlap_object, min_object
                gt_scene = gt_scene.random_subscene(
                    max_camera, min_overlap_object=self.config['min_overlap_object'], min_object=self.config['min_object'])

            if not gt_scene['camera']:
                return None, None

        if self.config['input_dir'] is None:
            input_scene = gt_scene.copy()
        else:
            # sample input_scene according to gt_scene
            input_scene = input_scene.subscene(gt_scene['camera'].keys())

        return self.set_world_coordinate_frame(input_scene, gt_scene)

    def get_data(self, input_scene, gt_scene):
        raise NotImplementedError

    def __getitem__(self, idx):
        input_scene, gt_scene = self.get_scene(idx)
        if input_scene is None:
            if self.config['ids']:
                raise ValueError(f"Invalid scene {self.config['ids'][idx]}")
            return None, None
        return self.get_data(input_scene, gt_scene)

    def filter_invalid_scene(self):
        # load invalid scenes if exists
        filter_setting = {k: self.config[k] for k in [
            'dir', 'min_camera', 'min_object', 'min_overlap_object', 'min_object_area']}
        filter_key = json.dumps(filter_setting)
        invalid_scene_dir = os.path.join(self.config['input_dir'] or self.config['dir'], f"{self.split}_invalid.json")
        if os.path.exists(invalid_scene_dir):
            with open(invalid_scene_dir) as f:
                invalid_scene_config = json.load(f)
        else:
            invalid_scene_config = {}

        if filter_key in invalid_scene_config:
            # use saved invalid scenes
            invalid_scene_id = invalid_scene_config[filter_key]
            print(f"Using the specified {len(invalid_scene_id)} invalid scenes from {invalid_scene_dir}")
        else:
            # filter invalid scenes
            invalid_scene_id = []
            obj_num = {cat: 0 for cat in CategoryMapping.object_categories}
            for idx in tqdm(range(len(self)), desc="Filtering invalid scenes"):
                gt_scene = self.get_scene(idx)[1]
                if gt_scene is None or len(gt_scene['camera']) < self.config['min_camera']:
                    invalid_scene_id.append(self.scene_ids[idx])
                    tqdm.write(f"Scene {self.scene_ids[idx]} is invalid")
                    continue
                for obj3d in gt_scene['object'].values():
                    obj_num[obj3d['category']] += 1

            # print object statistics
            print("Object category statistics:")
            for cat in CategoryMapping.object_categories:
                print(f"{cat}: {obj_num[cat]}")

            # save invalid scenes
            invalid_scene_config[filter_key] = invalid_scene_id
            with open(invalid_scene_dir, 'w') as f:
                json.dump(invalid_scene_config, f, indent=4)

        # filter scene ids
        invalid_scene_id = set(invalid_scene_id)
        self.scene_ids = [scene_id for scene_id in self.scene_ids if scene_id not in invalid_scene_id]
        print(f"Got {len(self.scene_ids)} valid scenes")


class PoseProposalModule(MyLightningModule):

    def train_on_scene(self, input_scene, gt_scene):
        self(input_scene)

        loss = self.loss(input_scene, gt_scene)
        log_dict = self.loss.compute_metrics()

        log_dict = {f"train/{k}": v for k, v in log_dict.items()}
        log_dict['global_step'] = self.global_step
        self.logger.experiment.log(log_dict)

        return loss

    def validation_step(self, batch, batch_idx):
        self.eval_batch(batch, self.val_metrics)

    def on_validation_epoch_end(self):
        if self.global_step > 0:
            log_dict = {f"val/{k}": v.compute() for k, v in self.val_metrics.items() if v._update_called}
            log_dict['global_step'], log_dict['epoch'] = self.global_step, self.current_epoch
            self.logger.experiment.log(log_dict)
        for m in self.val_metrics.values():
            m.reset()

    def test_step(self, batch, batch_idx):
        self.eval_batch(batch, self.test_metrics)

    def on_test_end(self):
        test_metrics = {k: v.compute() for k, v in self.test_metrics.items() if v._update_called}
        table = wandb.Table(columns=list(test_metrics.keys()), data=[list(test_metrics.values())])
        self.logger.experiment.log({'test/evaluation/metrics': table})
        test_metrics = {f"test/evaluation/{k}": v for k, v in test_metrics.items()}
        wandb.summary.update(test_metrics)


class SceneEstimationModule(MyLightningModule):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.test_metrics = nn.ModuleDict({k: torchmetrics.MeanMetric() for k in (
            'known_psnr', 'known_masked_psnr', 'known_ssim', 'known_lpips',
            'novel_psnr', 'novel_masked_psnr', 'novel_ssim', 'novel_lpips',
            'camera_translation', 'camera_rotation', 'scene_bdb3d_iou'
        )})
        self.bdb3d_info = {'gt_annos': [], 'dt_annos': [], 'metric': [0.15, 0.25]}
        self.test_metrics['affinity_AP'] = torchmetrics.classification.BinaryAveragePrecision()
        self.lpips = torchmetrics.image.lpip.LearnedPerceptualImagePatchSimilarity(net_type='vgg')

    def save_scene_result(self, est_scene):
        if not self.config['test']['save_results']:
            return
        wandb_logger = self.logger.experiment
        est_scene_np = est_scene.numpy() if est_scene.backend is torch else est_scene.copy()

        # save scene
        output_scene_root = os.path.join(wandb_logger.config['args']['output_dir'], wandb_logger.config['args']['id'], 'scene')
        output_scene_dir = os.path.join(output_scene_root, est_scene['uid'])
        shutil.rmtree(output_scene_dir, ignore_errors=True)
        data = est_scene_np.data_dict(
            keys=['uid', 'affinity', 'origin_camera_id'],
            camera={'keys': ['cam2world_mat', 'K', 'height', 'width', 'image', 'object', 'pitch', 'roll'],
                    'image': {'keys': ['color', 'depth', 'alpha']}, 'object': {'keys': [
                        'bdb2d', 'category', 'category_id', 'segmentation', 'area', 'orientation_score', 'orientation', 'center_depth', 'size', 'offset']}},
            object={'keys': {'rotation_mat', 'center', 'category_id', 'category', 'size', 'category_onehot', 'score', 'latent_code'}}
        )
        Scene(data).save(output_scene_dir)

    def gt_object_association_graph(self, est_scene_np, gt_scene_np):
        # match 2D object detection based on IoU
        obj2d_match = {}
        for cam_id in est_scene_np['camera'].keys():
            # calculate IoU between 2D object detection
            match_iou = {}
            est_obj2d_dict = est_scene_np['camera'][cam_id]['object']
            gt_obj2d_dict = gt_scene_np['camera'][cam_id]['object']
            for gt_obj2d_id, gt_obj2d in gt_obj2d_dict.items():
                for est_obj2d_id, est_obj2d in est_obj2d_dict.items():
                    iou = seg_iou(est_obj2d, gt_obj2d)
                    if iou > self.hparams.gt_match_iou_thres:
                        match_iou[(gt_obj2d_id, est_obj2d_id)] = iou

            # consider the best match first
            match_iou = dict(sorted(match_iou.items(), key=lambda item: item[1], reverse=True))
            matched_est_obj2d_ids = set()
            matched_gt_obj2d_ids = set()
            for gt_obj2d_id, est_obj2d_id in match_iou.keys():
                if est_obj2d_id in matched_est_obj2d_ids or gt_obj2d_id in matched_gt_obj2d_ids:
                    del match_iou[(gt_obj2d_id, est_obj2d_id)]

            # save node match
            obj2d_match.update({
                (cam_id, gt_obj2d_id): (cam_id, est_obj2d_id)
                for gt_obj2d_id, est_obj2d_id in match_iou.keys()
            })

        # generate gt object association graph from 2D object detection match
        gt_association_graph = gt_scene_np.object_association_graph
        gt_association_graph = gt_association_graph.subgraph(obj2d_match.keys())
        gt_association_graph = nx.relabel_nodes(gt_association_graph, obj2d_match)

        return gt_association_graph

    def visualize_scene(self, est_scene, gt_scene):
        if not self.config['test']['visualize_results']:
            return
        wandb_logger = self.logger.experiment
        visualize_gt = gt_scene['split'] != 'predict'

        novel_cam_id = set(cam_id for cam_id, cam in est_scene['camera'].items() if 'object' not in cam)
        est_scene = est_scene.subscene(set(est_scene['camera'].keys()) - novel_cam_id)
        gt_scene = gt_scene.subscene(est_scene['camera'].keys())
        est_scene_np, gt_scene_np = est_scene.numpy(), gt_scene.numpy()

        output_scene_root = os.path.join(wandb_logger.config['args']['output_dir'], wandb_logger.config['args']['id'], 'scene')
        output_scene_dir = os.path.join(output_scene_root, est_scene_np['uid'])
        os.makedirs(output_scene_dir, exist_ok=True)

        # visulize scene with ground truth
        top_view_settings = dict(
            scene_camera=dict(eye=dict(x=0, y=0, z=1.), up=dict(x=0, y=1, z=0), projection=dict(type='orthographic')),
            scene=dict(zaxis=dict(title='', showticklabels=False, tickangle=0))
        )
        if visualize_gt:
            est_scene_vis = SceneVisualizer(est_scene_np)
            gt_scene_vis = SceneVisualizer(gt_scene_np)
            fig = est_scene_vis.scene_vis(reference_scene_vis=gt_scene_vis)
            fig.write_image(os.path.join(output_scene_dir, 'scene_vis-compare.png'), scale=2)
            fig.update_layout(**top_view_settings)
            fig.write_image(os.path.join(output_scene_dir, 'scene_vis-top_view-compare.png'), scale=2)

        # visualize estimation scene
        est_scene_vis = SceneVisualizer(est_scene_np)
        fig = est_scene_vis.scene_vis()
        fig.write_image(os.path.join(output_scene_dir, 'scene_vis.png'), scale=2)
        fig.update_layout(**top_view_settings)
        fig.write_image(os.path.join(output_scene_dir, 'scene_vis-top_view.png'), scale=2)

        # visualize ground truth scene
        if visualize_gt:
            gt_scene_vis = SceneVisualizer(gt_scene_np) if visualize_gt else None
            fig = gt_scene_vis.scene_vis()
            fig.write_image(os.path.join(output_scene_dir, 'scene_vis-gt.png'), scale=2)
            fig.update_layout(**top_view_settings)
            fig.write_image(os.path.join(output_scene_dir, 'scene_vis-top_view-gt.png'), scale=2)

        # visulize object association graph
        gt_scene_vis = SceneVisualizer(gt_scene_np.subscene(est_scene_np['camera'].keys()))
        association_graph_img = est_scene_vis.object_association_graph(get_image=True, image_scene=gt_scene_np)
        Image.fromarray(association_graph_img).save(os.path.join(output_scene_dir, 'object_association_graph.jpg'))
        if visualize_gt:
            gt_association_graph_img = gt_scene_vis.object_association_graph(get_image=True)
            Image.fromarray(gt_association_graph_img).save(os.path.join(output_scene_dir, 'object_association_graph-gt.jpg'))
            association_graph_img = image_grid([association_graph_img, gt_association_graph_img], rows=1)
            Image.fromarray(association_graph_img).save(os.path.join(output_scene_dir, 'object_association_graph-compare.jpg'))

        # visualize object association graph for each camera pair
        output_association_dir = os.path.join(output_scene_dir, 'object_association_graph')
        os.makedirs(output_association_dir, exist_ok=True)
        for cam1_id in est_scene_np['camera'].keys():
            for cam2_id in est_scene_np['camera'].keys():
                if cam1_id <= cam2_id:
                    continue
                est_subscene_np = est_scene_np.subscene([cam1_id, cam2_id])
                gt_subscene_np = gt_scene_np.subscene([cam1_id, cam2_id])
                est_subscene_vis = SceneVisualizer(est_subscene_np)
                gt_subscene_vis = SceneVisualizer(gt_subscene_np)
                association_graph_img = est_subscene_vis.object_association_graph(get_image=True, image_scene=gt_subscene_np)
                if visualize_gt:
                    gt_association_graph_img = gt_subscene_vis.object_association_graph(get_image=True)
                    association_graph_img = image_grid([association_graph_img, gt_association_graph_img], rows=1)
                Image.fromarray(association_graph_img).save(os.path.join(output_association_dir, f"{cam1_id}_{cam2_id}.jpg"))

    def eval_scene(self, est_scene, gt_scene):
        if not self.config['test']['evaluate_results']:
            return
        wandb_logger = self.logger.experiment
        output_scene_root = os.path.join(wandb_logger.config['args']['output_dir'], wandb_logger.config['args']['id'], 'scene')
        output_scene_dir = os.path.join(output_scene_root, est_scene['uid'])
        os.makedirs(output_scene_dir, exist_ok=True)
        test_sample_row = {'uid': gt_scene['uid'], 'cameras': len(gt_scene['camera'])}

        novel_cam_id = set(cam_id for cam_id, cam in est_scene['camera'].items() if 'object' not in cam)
        est_scene_novel = est_scene.subscene(novel_cam_id)
        est_scene = est_scene.subscene(set(est_scene['camera'].keys()) - novel_cam_id)
        gt_scene_novel = gt_scene.subscene(novel_cam_id)
        est_scene_novel_np, gt_scene_novel_np = est_scene_novel.numpy(), gt_scene_novel.numpy()
        gt_scene = gt_scene.subscene(est_scene['camera'].keys())
        est_scene_np, gt_scene_np = est_scene.numpy(), gt_scene.numpy()

        # evaluate camera related metrics
        scene_metrics = defaultdict(list)

        # evaluate image quality
        for prefix, est_scn, gt_scn in (('novel', est_scene_novel, gt_scene_novel), ('known', est_scene, gt_scene)):
            for camera_id, est_camera in est_scn['camera'].items():
                if 'masked_color' not in gt_scn['camera'][camera_id]['image']:
                    continue
                est_color = est_camera['image']['color']
                gt_color = gt_scn['camera'][camera_id]['image']['masked_color']
                psnr = torchmetrics.functional.peak_signal_noise_ratio(est_color, gt_color, data_range=1.0).item()
                psnr = min(psnr, 100)

                mask = gt_scn['camera'][camera_id]['image']['instance_segmap'] >= 0
                est_color_masked = est_color[mask]
                gt_color_masked = gt_color[mask]
                masked_psnr = torchmetrics.functional.peak_signal_noise_ratio(est_color_masked, gt_color_masked, data_range=1.0).item()
                masked_psnr = min(masked_psnr, 100)

                est_color = est_color.permute(2, 0, 1).unsqueeze(0)
                gt_color = gt_color.permute(2, 0, 1).unsqueeze(0)
                ssim = torchmetrics.functional.structural_similarity_index_measure(est_color, gt_color, data_range=1.0).item()
                lpips = self.lpips(est_color, gt_color).item()
                scene_metrics[f"{prefix}_psnr"].append(psnr)
                scene_metrics[f"{prefix}_masked_psnr"].append(masked_psnr)
                scene_metrics[f"{prefix}_ssim"].append(ssim)
                scene_metrics[f"{prefix}_lpips"].append(lpips)
                self.test_metrics[f"{prefix}_psnr"].update(psnr)
                self.test_metrics[f"{prefix}_masked_psnr"].update(masked_psnr)
                self.test_metrics[f"{prefix}_ssim"].update(ssim)
                self.test_metrics[f"{prefix}_lpips"].update(lpips)

        # evaluate camera pose
        with suppress(GetFailedError):  # skip if camera pose is not available
            for camera_id, est_camera_np in est_scene_np['camera'].items():
                if camera_id != est_scene_np['origin_camera_id']:
                    gt_camera_np = gt_scene_np['camera'][camera_id]
                    translation_error = np.linalg.norm(est_camera_np['position'] - gt_camera_np['position'])
                    scene_metrics['camera_translation'].append(translation_error)
                    self.test_metrics['camera_translation'].update(translation_error)

                    est_relative_rotation_mat = est_camera_np['rotation_mat'] @ est_scene_np['camera'][est_scene_np['origin_camera_id']]['rotation_mat'].T
                    gt_relative_rotation_mat = gt_camera_np['rotation_mat'] @ gt_scene_np['camera'][est_scene_np['origin_camera_id']]['rotation_mat'].T
                    rotation_error = rotation_mat_dist(est_relative_rotation_mat, gt_relative_rotation_mat)
                    scene_metrics['camera_rotation'].append(rotation_error)
                    self.test_metrics['camera_rotation'].update(rotation_error)

        scene_metrics = {k: np.mean(v) for k, v in scene_metrics.items()}
        test_sample_row.update(scene_metrics)

        # evaluate object association
        if est_scene.get('affinity', False) and gt_scene.get('affinity', False):
            # generate gt affinity matrices from gt association graph
            gt_affinity = {}
            for key, est_affinity in est_scene['affinity'].items():
                gt_affinity[key] = torch.zeros(est_affinity.shape, dtype=np.bool, device=self.device)
            obj_id2idx = {
                cam_id: {obj_id: idx for idx, obj_id in enumerate(cam['object'].keys())}
                for cam_id, cam in est_scene['camera'].items()
            }
            gt_association_graph = self.gt_object_association_graph(est_scene_np, gt_scene_np)
            for (cam1_id, obj1_id), (cam2_id, obj2_id) in gt_association_graph.edges:
                if (cam1_id, cam2_id) in gt_affinity:
                    gt_affinity[(cam1_id, cam2_id)][obj_id2idx[cam1_id][obj1_id], obj_id2idx[cam2_id][obj2_id]] = True
                else:
                    gt_affinity[(cam2_id, cam1_id)][obj_id2idx[cam2_id][obj2_id], obj_id2idx[cam1_id][obj1_id]] = True

            # generate est affinity matrices from est association graph
            est_association_graph = est_scene_np.object_association_graph
            est_affinity = {k: est_scene['affinity'][k].clone().float() for k in est_scene['affinity'].keys()}
            for (cam1_id, obj1_id), (cam2_id, obj2_id) in est_association_graph.edges:
                if (cam1_id, cam2_id) in est_affinity:
                    est_affinity[(cam1_id, cam2_id)][obj_id2idx[cam1_id][obj1_id], obj_id2idx[cam2_id][obj2_id]] *= 2
                else:
                    est_affinity[(cam2_id, cam1_id)][obj_id2idx[cam2_id][obj2_id], obj_id2idx[cam1_id][obj1_id]] *= 2
            est_affinity = {k: v / 2 for k, v in est_affinity.items()}  # based on Associative3D

            # evaluate association metrics
            est_affinity = torch.cat([v.flatten() for v in est_affinity.values()])
            gt_affinity = torch.cat([v.flatten() for v in gt_affinity.values()])
            self.test_metrics['affinity_AP'].update(est_affinity, gt_affinity)
            test_sample_row['affinity_AP'] = torchmetrics.functional.average_precision(est_affinity, gt_affinity, 'binary').item()

        # evaluate object detection
        if 'object' in est_scene_np and 'object' in gt_scene_np:
            with suppress(ImportError):
                from mmdet3d.core.bbox import DepthInstance3DBoxes
                dt_anno = defaultdict(list)
                for est_obj in est_scene_np['object'].values():
                    if est_obj['category'] == 'bookshelf':
                        continue
                    dt_anno['labels_3d'].append(est_obj['category_id'])
                    dt_anno['boxes_3d'].append(est_obj.mmdet3d())
                    dt_anno['scores_3d'].append(est_obj['score'])
                dt_anno = {k: torch.from_numpy(np.stack(v)) for k, v in dt_anno.items()}
                dt_anno['boxes_3d'] = DepthInstance3DBoxes(dt_anno['boxes_3d'], with_yaw=True)
                self.bdb3d_info['dt_annos'].append(dt_anno)

                gt_anno = {'gt_boxes_upright_depth': [], 'class': []}
                for gt_obj in gt_scene_np['object'].values():
                    if gt_obj['category'] == 'bookshelf':
                        continue
                    gt_anno['gt_boxes_upright_depth'].append(gt_obj.mmdet3d())
                    gt_anno['class'].append(gt_obj['category_id'])
                gt_anno = {k: np.stack(v) for k, v in gt_anno.items()}
                gt_anno['gt_num'] = len(gt_scene_np['object'])
                self.bdb3d_info['gt_annos'].append(gt_anno)

        # visualize output images
        for prefix, est_scn, gt_scn in (('novel', est_scene_novel_np, gt_scene_novel_np), ('known', est_scene_np, gt_scene_np)):
            est_color, gt_color = [], []
            for camera_id, est_camera in est_scn['camera'].items():
                if 'masked_color' not in gt_scn['camera'][camera_id]['image']:
                    continue
                est_color.append(image_float_to_uint8(est_camera['image']['color']))
                gt_color.append(image_float_to_uint8(gt_scn['camera'][camera_id]['image']['masked_color']))
            if not est_color:
                continue
            combined_img = image_grid([est_color, gt_color], padding=2, background_color=(128, 128, 128))
            img_dir = os.path.join(output_scene_dir, f'{prefix}-color.jpg')
            Image.fromarray(combined_img).save(img_dir)
            test_sample_row[f'{prefix}_color'] = wandb.Image(img_dir)

        if not hasattr(self, 'test_sample_table'):
            self.test_sample_table = wandb.Table(columns=list(test_sample_row.keys()))
        self.test_sample_table.add_data(*test_sample_row.values())

    def on_test_start(self):
        if self.config['test']['skip_prediction'] or not self.config['test']['save_results']:
            return
        # save dataset info if not exists
        wandb_logger = self.logger.experiment
        output_scene_root = os.path.join(wandb_logger.config['args']['output_dir'], wandb_logger.config['args']['id'], 'scene')
        os.makedirs(output_scene_root, exist_ok=True)
        for category_config in ('object_category2future', 'object_categories'):
            if not os.path.exists(os.path.join(output_scene_root, f"{category_config}.yaml")):
                shutil.copy(os.path.join(self.config['dataset']['dir'], f"{category_config}.yaml"), output_scene_root)

    def on_test_end(self):
        if not self.config['test']['evaluate_results']:
            return

        # collect test metrics
        test_metrics = {k: v.compute() for k, v in self.test_metrics.items() if v._update_called}
        if self.bdb3d_info['gt_annos']:
            from mmdet3d.core.bbox.structures import Box3DMode, DepthInstance3DBoxes
            from mmdet3d.core.evaluation.indoor_eval import indoor_eval
            ret_value = indoor_eval(
                **self.bdb3d_info,
                label2cat={i: c for i, c in enumerate(CategoryMapping.object_categories) if c != 'bookshelf'},
                box_type_3d=DepthInstance3DBoxes,
                box_mode_3d=Box3DMode.DEPTH
            )
            ret_value = {f"det3d_{k}": v for k, v in ret_value.items()}
            test_metrics.update(ret_value)

        # log test metrics as table
        table = wandb.Table(columns=list(test_metrics.keys()), data=[list(test_metrics.values())])
        self.logger.experiment.log({'test/evaluation/metrics': table})

        # log test metrics as summary
        test_metrics = {f"test/evaluation/{k}": v for k, v in test_metrics.items()}
        wandb.summary.update(test_metrics)
        self.logger.experiment.log({'test/samples': self.test_sample_table})

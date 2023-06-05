from .scene_proposal import SceneProposal, SceneProposalDataset
from .scene_optimization import SceneOptimization, SceneOptimizationDataset
from .base import SceneEstimationModule
import yaml
from utils.general_utils import recursive_merge_dict
from utils.dataset import Scene, Camera
import torch
from external.shapenet_renderer_utils import get_archimedean_spiral, look_at
import numpy as np
import os
import copy
from PIL import Image
from utils.visualize_utils import image_float_to_uint8


class SceneUnderstandingDataset(SceneProposalDataset, SceneOptimizationDataset):
    def get_data(self, input_scene, gt_scene):
        # add data from SceneOptimizationDataset
        # including encoder input without zoom-out
        input_data, gt_data = SceneOptimizationDataset.get_data(
            self,
            gt_scene.subscene(list(input_scene['camera'].keys())) if self.config['gt_affinity'] else input_scene,
            gt_scene
        )

        # add data from SceneProposalDataset
        # copy input_scene to avoid modifying 2D bounding boxes when zooming out
        input_data_prop, gt_data_prop = SceneProposalDataset.get_data(self, input_scene.copy(), gt_scene)
        input_data = recursive_merge_dict(input_data, input_data_prop)
        gt_data = recursive_merge_dict(gt_data, gt_data_prop)

        return input_data, gt_data 

class SceneUnderstanding(SceneEstimationModule):
    dataset_cls = SceneUnderstandingDataset

    def __init__(self, scene_proposal, scene_optimization, **kwargs):
        super().__init__(**kwargs)
        self.save_hyperparameters()

        # initialize submodules with from given config files
        for Model, overwrite_config, attr_name in (
                (SceneProposal, scene_proposal, 'scene_proposal'),
                (SceneOptimization, scene_optimization, 'scene_optimization')):
            config_dir = overwrite_config.pop('config_dir')
            print(f"Reading config file from {config_dir} for {Model.__name__}")
            with open(config_dir, 'r') as f:
                config = yaml.load(f, Loader=yaml.FullLoader)
            config = recursive_merge_dict(config['model'], overwrite_config)
            config['dataset'].update(self.config['dataset'])

            setattr(self, attr_name, Model(**config))

        # use the same dataset config as scene proposal
        dataset_config = self.scene_proposal.config['dataset'].copy()
        dataset_config.update(self.config.get('dataset', {}))
        dataset_config['input_images']['nerf_enc_input'] = self.scene_optimization.config['dataset']['input_images']
        self.config['dataset'] = dataset_config
        if self.config['test']['skip_prediction']:
            dataset_config['input_images'] = None

        # disable camera pose optimization if gt camera pose is given
        if self.config['dataset']['gt_camera_pose']:
            for key in ('camera_rotation_6d', 'camera_position'):
                self.scene_optimization.config['test']['lr'][key] = 'override_with_null'
            self.scene_optimization.loss.disable_loss('camera_pitch_constraint')
            self.scene_optimization.loss.disable_loss('camera_roll_constraint')
            if self.config['dataset']['gt_object_pose']:
                assert self.config['dataset']['gt_affinity'], 'gt_affinity must be enabled if gt_camera_pose and gt_object_pose is enabled'
                for key in ('object_rotation_6d', 'object_center', 'object_size'):
                    self.scene_optimization.config['test']['lr'][key] = 'override_with_null'
        elif self.config['dataset']['gt_object_pose']:
            raise NotImplementedError('gt_object_pose cannot be used without gt_camera_pose')

    def forward(self, est_scene, gt_scene):
        # camera/object proposal
        get_pitch_roll = not self.config['dataset']['gt_pitch_roll'] and not self.config['dataset']['gt_camera_pose']
        get_camera_yaw = not self.config['dataset']['gt_camera_yaw'] and not self.config['dataset']['gt_camera_pose']
        self.scene_proposal.camera_object_proposal(
            est_scene, gt_scene, get_pitch_roll, get_camera_yaw,
            not self.config['dataset']['gt_object_pose'], not self.config['dataset']['gt_affinity'],
        )

        # scene proposal + optimization
        self.scene_proposal.scene_optimization = self.scene_optimization
        est_scene = self.scene_proposal.generate_scene(est_scene, gt_scene)

        # render final scene into color images including novel views
        camera_dict = est_scene['camera'].copy()
        camera_dict.update({cam_id: cam for cam_id, cam in gt_scene['camera'].items() if cam_id not in camera_dict})
        camera_dict = self.scene_optimization.render_scene(est_scene, camera_dict) #
        for cam_id, cam in camera_dict.items():
            if cam_id in est_scene['camera']:
                est_scene['camera'][cam_id]['image'].update(cam['image'])
            else:
                est_scene['camera'][cam_id] = Camera(cam, backend=torch, device=self.device)

        # save estimated scene
        self.save_scene_result(est_scene)

        return est_scene

    def test_step(self, batch, batch_idx):
        input_data, gt_data = batch
        est_scene, gt_scene = Scene(input_data, backend=torch, device=self.device), Scene(gt_data, backend=torch, device=self.device)
        
        est_scene_copy = est_scene.clone()
        for camid, cam in est_scene_copy['camera'].items():
            for objid, obj in cam['object'].items():
                if len(obj['camera']) == 1:
                    est_scene['camera'][camid]['object'].pop(objid)

        if not self.config['test']['skip_prediction']:
            est_scene = self(est_scene, gt_scene)

        # evaluate scene
        self.eval_scene(est_scene, gt_scene) 

        # visualize scene
        self.visualize_scene(est_scene, gt_scene)

    def predict_step(self, batch, batch_idx):
        self.test_step(batch, batch_idx)

    def render_novel(self, est_scene, camera_poses, folder):
        empty_camera = next(iter(est_scene['camera'].values())).copy()
        
        #render novel views
        camera_dict = {}
        for cam_id, camera_pose in enumerate(camera_poses):
            camera = empty_camera.copy()
            camera['cam2world_mat'] = camera_pose
            camera_dict[cam_id] = camera
        camera_dict = self.scene_optimization.render_scene(est_scene, camera_dict)

        # save generated novel views
        wandb_logger = self.logger.experiment
        output_scene_root = os.path.join(wandb_logger.config['args']['output_dir'], wandb_logger.config['args']['id'], 'scene')
        output_scene_dir = os.path.join(output_scene_root, est_scene['uid'])
        output_frame_dir = os.path.join(output_scene_dir, folder)
        os.makedirs(output_frame_dir, exist_ok=True)
        for cam_id, cam in camera_dict.items():
            frame = image_float_to_uint8(Camera(cam)['image']['color'])
            Image.fromarray(frame).save(os.path.join(output_frame_dir, f"{cam_id:04d}.jpg"))

    def visualize_scene(self, est_scene, gt_scene):
        if not self.config['test']['visualize_results']:
            return

        super().visualize_scene(est_scene, gt_scene)
        # if gt_scene['split'] != 'predict':
        #     return

        ids = list(est_scene['camera'].keys())
        # divide est_scene_camera
        for id in ids:
            if len(est_scene['camera'][id]) > 1:
                continue
            est_scene['camera'].pop(id)

        est_scene_np = est_scene.numpy()
        # generate scene info
        scene_points = [obj['center'] for obj in est_scene_np['object'].values()]
        scene_center = np.mean(scene_points, axis=0)
        scene_radius = np.max(np.linalg.norm(scene_points - scene_center, axis=1)) * 2
        camera_points = [cam['position'] for cam in est_scene_np['camera'].values()]
        camera_center = np.mean(camera_points, axis=0)
        camera_radius = np.max(np.linalg.norm(camera_points - camera_center, axis=1)) * 2

        # generate side-view camera poses
        camera_locations = np.random.normal(size=(self.config['test']['num_side_view'], 3)) * camera_radius
        camera_locations += camera_center
        camera_poses = look_at(camera_locations, scene_center)
        self.render_novel(est_scene, camera_poses, 'side-view')

        # generate spiral-view camera poses
        camera_locations = get_archimedean_spiral(self.config['test']['num_spiral_view'], scene_radius)
        camera_locations += scene_center
        camera_poses = look_at(camera_locations, scene_center)
        self.render_novel(est_scene, camera_poses, 'spiral-view')

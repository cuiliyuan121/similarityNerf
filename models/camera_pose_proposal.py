import torch
from torch import optim, nn
import pytorch_lightning as pl
from .base import Front3DDataset, PoseProposalModule
from utils.torch_utils import LossModule, get_resnet
from utils.transform import WORLD_FRAME_TO_TOTAL3D, CAMERA_FRAME_TO_TOTAL3D
from utils.dataset import parameterize_camera, Scene
from scipy.spatial.transform import Rotation
import torchmetrics
import copy
import numpy as np
from utils.visualize_utils import image_grid
import wandb
import matplotlib.pyplot as plt
from scipy.special import softmax


class CameraPoseProposalDataset(Front3DDataset):
    def get_data(self, input_scene, gt_scene):
        # apply transform on image data
        input_images = self.config['input_images']['cam_prop_input'] if isinstance(self.config['input_images'], dict) else self.config['input_images']
        if input_images:
            for camera in input_scene['camera'].values():
                camera['image']['cam_prop_input'] = camera['image'].preprocess_image(input_images)

        if self.split != 'predict':
            # convert camera rotation to euler angles
            for camera in gt_scene['camera'].values():
                parameterize_camera(camera)

            # generate relative yaw gt
            cam_yaw = np.array([cam['yaw'] for cam in gt_scene['camera'].values()], dtype=np.float32)
            cam_yaw_mat = np.repeat(cam_yaw[..., np.newaxis], len(cam_yaw), axis=-1)
            relative_yaw = (cam_yaw_mat - cam_yaw_mat.T) % (2 * np.pi)
            gt_scene['relative_yaw'] = relative_yaw

        # generate relative yaw query
        relative_yaw_query = np.linspace(0, 2 * np.pi, self.config['relative_yaw_query'] + 1)[:-1]
        relative_yaw_query = np.tile(relative_yaw_query, (len(input_scene['camera']), len(input_scene['camera']), 1))
        if self.split == 'train':
            relative_yaw_query = (relative_yaw[..., np.newaxis] + relative_yaw_query) % (2 * np.pi)
        input_scene['relative_yaw_query'] = relative_yaw_query

        # get data as a dict
        input_data = input_scene.data_dict(keys=['uid', 'relative_yaw_query'], camera={
            'keys': ['height', 'width', 'K'], 'image': {'keys': ['cam_prop_input']}})

        gt_data = gt_scene.data_dict(keys=['uid', 'relative_yaw'], camera={
            'keys': ['pitch', 'roll'], 'image': {'keys': ['color']}})

        return input_data, gt_data


class CameraPoseProposalNet(pl.LightningModule):
    def __init__(self, backbone, n_layers, W, positional_encoding, pretrained=False, input_images=None):
        super().__init__()
        self.save_hyperparameters()

        # initialize feature embedding
        self.resnet, n_inputs = get_resnet(backbone, pretrained=pretrained, input_images=input_images)

        # initialize MLP head
        dim_pe_input = positional_encoding * 2 if positional_encoding else 1
        self.out_dims = {'pitch_roll': 2, 'relative_yaw': 1}
        for mlp_name, final_out_dim in self.out_dims.items():
            if final_out_dim is None:
                continue
            layers = []
            in_dim, out_dim = n_inputs * 2 + dim_pe_input if mlp_name == 'relative_yaw' else n_inputs, W
            for i in range(n_layers):
                if i == n_layers - 1:
                    out_dim = final_out_dim
                layers.append(nn.Linear(in_dim, out_dim))
                if i < n_layers - 1:
                    layers.append(nn.LeakyReLU(0.2, True))
                in_dim = out_dim
            mlp = nn.Sequential(*layers)
            setattr(self, f"{mlp_name}_mlp", mlp)

    def forward(self, image):
        image_feature = self.resnet(image)
        results = self.pitch_roll_mlp(image_feature)
        return {'pitch': results[:, 0], 'roll': results[:, 1], 'image_feature': image_feature}

    def positional_encoding(self, x):
        if not self.hparams.positional_encoding:
            return x.unsqueeze(-1)
        pe_coeff = 2 ** torch.arange(self.hparams.positional_encoding, device=x.device)
        embed = x[..., None] * pe_coeff
        return torch.cat((embed.sin(), embed.cos()), dim=-1)

    def relative_yaw_score(self, image_feature1, image_feature2, relative_yaw):
        image_feature1 = image_feature1.expand(len(relative_yaw), -1)
        image_feature2 = image_feature2.expand(len(relative_yaw), -1)
        relative_yaw = self.positional_encoding(relative_yaw)
        image_feature = torch.cat([image_feature1, image_feature2, relative_yaw], dim=-1)
        return self.relative_yaw_mlp(image_feature)

    def calculate_camera_pose_for_scene(self, scene, get_pose=True, get_relative_yaw=True):
        if not get_pose and not get_relative_yaw:
            return

        # estimate camera pose and extract image feature
        batched_image = torch.stack([camera['image']['cam_prop_input'] for camera in scene['camera'].values()])
        est_pose = self(batched_image)
        keys = []
        if get_pose:
            keys.extend(['pitch', 'roll'])
        if get_relative_yaw:
            keys.extend(['image_feature'])
        for camera_idx, camera in enumerate(scene['camera'].values()):
            camera.update({k: est_pose[k][camera_idx] for k in keys})

        # compute relative yaw score
        if get_relative_yaw:
            relative_yaw_score = []
            for i_cam, cam1 in enumerate(scene['camera'].values()):
                relative_yaw_score.append([])
                for j_cam, cam2 in enumerate(scene['camera'].values()):
                    relative_yaw_query = scene['relative_yaw_query'][i_cam, j_cam]
                    relative_yaw_score[-1].append(self.relative_yaw_score(
                        cam1['image_feature'], cam2['image_feature'], relative_yaw_query).squeeze(-1))
                relative_yaw_score[-1] = torch.stack(relative_yaw_score[-1], dim=0)
            scene['relative_yaw_score'] = torch.stack(relative_yaw_score, dim=0)
            relative_yaw_idx = torch.argmax(scene['relative_yaw_score'], dim=-1)
            scene['relative_yaw'] = torch.gather(scene['relative_yaw_query'], 2, relative_yaw_idx.unsqueeze(-1)).squeeze(-1)
            scene['relative_yaw_prob'] = torch.softmax(scene['relative_yaw_score'], dim=-1)


class CameraPoseProposalLoss(LossModule):
    def __init__(self, func=None, weight=None, **kwargs):
        assert 'reg' in func and 'pitch' in weight and 'roll' in weight
        super().__init__(func, weight, **kwargs)

    def compute_loss(self, est_scene, gt_scene):
        loss = 0.

        # compute pitch and roll loss
        for est_camera, gt_camera in zip(est_scene['camera'].values(), gt_scene['camera'].values()):
            for key in ['pitch', 'roll']:
                if self.weight[key] is None:
                    continue
                sub_loss = self.func['reg'](est_camera[key], gt_camera[key])
                self.metrics[f"{key}_loss"].update(sub_loss)
                if self.weight[key]:
                    loss += self.weight[key] * sub_loss / len(est_scene['camera'])

        # compute relative yaw loss
        if self.weight['relative_yaw'] is not None:
            log_prob = torch.log_softmax(est_scene['relative_yaw_score'], dim=-1)
            relative_yaw_loss = -torch.mean(log_prob[..., 0])
            self.metrics['relative_yaw_loss'].update(relative_yaw_loss)
        if self.weight['relative_yaw']:
            loss += self.weight['relative_yaw'] * relative_yaw_loss

        return loss


class CameraPoseProposal(PoseProposalModule):
    dataset_cls = CameraPoseProposalDataset

    def __init__(self, camera_pose_proposal, input_images=None, loss=None, **kwargs):
        super().__init__(**kwargs)
        self.save_hyperparameters()

        # set input images
        self.config['dataset']['input_images'] = input_images
        camera_pose_proposal['input_images'] = input_images

        # initialize camera pose proposal network
        self.camera_pose_proposal = CameraPoseProposalNet(**camera_pose_proposal)

        # initialize loss
        if loss:
            self.loss = CameraPoseProposalLoss(**loss)

        # define metrics
        self.val_metrics = nn.ModuleDict({k: torchmetrics.MeanAbsoluteError() for k in ('pitch', 'roll')})
        self.val_metrics['relative_yaw'] = torchmetrics.MeanMetric()
        self.test_metrics = copy.deepcopy(self.val_metrics)

    def configure_optimizers(self):
        params = [{'params': self.camera_pose_proposal.parameters(), 'lr': self.config['train']['lr'], 'name': 'camera_pose_proposal'}]
        optimizer = optim.AdamW(params)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, **self.config['train']['lr_scheduler'])
        return [optimizer], [lr_scheduler]

    def forward(self, scene, **kwargs):
        self.camera_pose_proposal.calculate_camera_pose_for_scene(scene, **kwargs)

    def training_step(self, batch, batch_idx):
        input_data, gt_data = batch
        input_scene = Scene(input_data, backend=torch, device=self.device)
        gt_scene = Scene(gt_data, backend=torch, device=self.device)

        loss = super().train_on_scene(input_scene, gt_scene)

        if input_scene['if_log']:
            input_scene_np, gt_scene_np = input_scene.numpy(), gt_scene.numpy()

            # log sample input image
            sample_cams = list(gt_scene_np['camera'].values())[:2]
            input_images = [cam['image']['color'] for cam in sample_cams]
            input_image = image_grid(input_images, rows=1)

            # log estimated relative camera yaw probability distribution
            relative_yaw_query = np.rad2deg(input_scene_np['relative_yaw_query'][0, 1])
            relative_yaw_score = input_scene_np['relative_yaw_score'][0, 1]
            relative_yaw_prob = softmax(relative_yaw_score)
            relative_yaw_error = np.sort(relative_yaw_query)
            plt.plot(relative_yaw_error, relative_yaw_prob)
            plt.xlabel("relative yaw error (deg)")
            plt.ylabel(relative_yaw_prob)
            relative_yaw_prob_img = wandb.Image(plt)
            plt.close()

            log_dict = {'global_step': self.global_step, 'input_image': wandb.Image(input_image), 'relative_yaw_prob': relative_yaw_prob_img}
            self.logger.experiment.log(log_dict)
        return loss

    def eval_batch(self, batch, metrics):
        input_data, gt_data = batch
        self(input_data)

        # evaluate camera pose
        for camera_est, camera_gt in zip(input_data['camera'].values(), gt_data['camera'].values()):
            for k in ('pitch', 'roll'):
                metrics[k].update(torch.rad2deg(camera_est[k]), torch.rad2deg(camera_gt[k]))

        # evaluate relative yaw
        relative_yaw_error = torch.abs(input_data['relative_yaw'] - gt_data['relative_yaw'])
        relative_yaw_error = torch.min(relative_yaw_error, 2 * np.pi - relative_yaw_error)
        metrics['relative_yaw'].update(torch.rad2deg(relative_yaw_error).flatten())

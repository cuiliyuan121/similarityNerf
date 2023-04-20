import wandb
import numpy as np
import torch
from torch import optim, nn
import pytorch_lightning as pl
from .base import Front3DDataset, PoseProposalModule
from utils.torch_utils import LossModule, get_resnet
from utils.dataset import CategoryMapping, Object2D, Object, Scene, object3d_from_prediction, Camera, parameterize_object2d, Affinity
from utils.transform import homotrans, cam2uv, bdb3d_corners, rotation_mat_dist
from utils.metrics import bdb3d_iou
from scipy.spatial.transform import Rotation
from collections import defaultdict
from external.total3d.relation_net import RelationNet, bdb2d_geometric_feature
import torchmetrics
import copy
from utils.visualize_utils import image_grid, affinity_heatmap
import networkx as nx


def affinity_from_id_for_scene(scene):
    affinity = {}
    for cam1_id, cam1 in scene['camera'].items():
        for cam2_id, cam2 in scene['camera'].items():
            if cam1_id == cam2_id or (cam2_id, cam1_id) in affinity:
                continue
            obj1_id = np.array(list(cam1['object'].keys()))
            obj1_id = np.repeat(obj1_id[..., np.newaxis], len(cam2['object']), axis=-1)
            obj2_id = np.array(list(cam2['object'].keys()))
            obj2_id = np.repeat(obj2_id[np.newaxis, ...], len(cam1['object']), axis=0)
            affinity[(cam1_id, cam2_id)] = obj1_id == obj2_id
    scene['affinity'] = Affinity(affinity, backend=scene.backend, device=scene.device)


class ObjectProposalDataset(Front3DDataset):
    def get_data(self, input_scene, gt_scene):
        if self.split != 'predict':
            # augment 2d bounding box if in training and using gt 2d bounding box as input
            if self.split == 'train' and self.config['input_dir'] is None:
                input_scene.add_noise_to_bdb2d(self.config['bdb2d_noise_std'])

            # generate object pose parameters according to input 2d bounding box
            parameterize_object2d(input_scene if self.split == 'train' else gt_scene, gt_scene)

        # crop images for each object
        zoom_out_scene = input_scene.copy()
        zoom_out_scene.zoom_out_bdb2d(self.config['zoom_out_ratio'])
        zoom_out_scene.crop_object_images()

        # apply transform on image data
        input_images = self.config['input_images']['obj_prop_input'] if isinstance(self.config['input_images'], dict) else self.config['input_images']
        if input_images:
            for input_cam, zoom_out_cam in zip(input_scene['camera'].values(), zoom_out_scene['camera'].values()):
                if 'object' not in input_cam:
                    continue
                for ref_obj2d, zoom_out_obj2d in zip(input_cam['object'].values(), zoom_out_cam['object'].values()):
                    ref_obj2d['camera'] = Camera({'image': {
                        'obj_prop_input': zoom_out_obj2d['camera']['image'].preprocess_image(input_images, self.config['resize_width'])}})

        # generate affinity matrix gt
        if self.split != 'predict':
            affinity_from_id_for_scene(gt_scene)

        # get data as a dict
        input_data = input_scene.data_dict(keys=['uid'], camera={'keys': [], 'object': {
            'keys': ['segmentation', 'area', 'category', 'category_id', 'bdb2d', 'score'], 'camera': {'keys': [], 'image': {'keys': ['obj_prop_input']}}}})
        gt_data = gt_scene.data_dict(keys=['uid', 'affinity'], camera={'object': {
            'keys': ['category', 'bdb2d', 'orientation', 'orientation_cls', 'center_depth', 'mean_masked_depth', 'center_depth_offset', 'size', 'size_scale', 'offset']}})

        return input_data, gt_data


class ObjectPoseProposalNet(pl.LightningModule):
    def __init__(self, backbone, n_layers, W, num_category, category_emdedding, orientation_bin, relation_net, dim_embedding, pretrained=False, input_images=None):
        super().__init__()
        self.save_hyperparameters()

        # initialize feature embedding
        self.resnet, n_inputs = get_resnet(backbone, pretrained=pretrained, input_images=input_images)

        # initialize relation net
        self.relation_net = RelationNet(**relation_net, a_feature_length=n_inputs)

        # initialize category embedding
        self.category_embedding = nn.Embedding(num_category, category_emdedding)

        # initialize MLP head
        Object.generate_orientation_bin(orientation_bin)
        self.out_dims = {'orientation_score': orientation_bin, 'center_depth': 1, 'size_scale': 3, 'offset': 2,
                         'embedding': dim_embedding}
        for mlp_name, final_out_dim in self.out_dims.items():
            if final_out_dim is None:
                continue
            layers = []
            in_dim, out_dim = n_inputs + category_emdedding, W
            for i in range(n_layers):
                if i == n_layers - 1:
                    out_dim = final_out_dim
                layers.append(nn.Linear(in_dim, out_dim))
                if i < n_layers - 1:
                    layers.append(nn.LeakyReLU(0.2, True))
                in_dim = out_dim
            mlp = nn.Sequential(*layers)
            setattr(self, f"{mlp_name}_mlp", mlp)

    def forward(self, image, category_id, g_features, split):
        a_features = self.resnet(image)

        rel_pair_counts = torch.cat([torch.tensor([0], device=split.device), torch.cumsum(torch.pow(split[:, 1] - split[:, 0], 2), 0)], 0)
        r_features = self.relation_net(a_features, g_features, split, rel_pair_counts)
        ar_features = torch.add(a_features, r_features)

        a_r_features_cat = torch.cat([ar_features, self.category_embedding(category_id)], dim=-1)

        results = {}
        for mlp_name, out_dim in self.out_dims.items():
            if out_dim is None:
                continue
            results[mlp_name] = getattr(self, f"{mlp_name}_mlp")(a_r_features_cat)
            if out_dim == 1:
                results[mlp_name] = results[mlp_name].squeeze(-1)

        return results

    def calculate_object_pose_for_scene(self, scene, get_pose=True, get_embedding=True):
        if not get_pose and not get_embedding:
            return
        keys = []
        if get_pose:
            keys.extend(['orientation_score', 'center_depth', 'size_scale', 'offset'])
        if get_embedding and self.out_dims['embedding']:
            keys.append('embedding')

        batched_input = defaultdict(list)
        category_ids = []
        split_begin = 0
        for camera in scene['camera'].values():
            for obj2d in camera['object'].values():
                batched_input['image'].append(obj2d['camera']['image']['obj_prop_input'])
                category_ids.append(obj2d['category_id'])
            g_features = bdb2d_geometric_feature(
                [obj2d['bdb2d'] for obj2d in camera['object'].values()],
                self.relation_net.d_g
            )
            batched_input['g_features'].append(g_features)
            batched_input['split'].append(torch.tensor(
                [split_begin, split_begin + len(camera['object'])], device=g_features.device))
            split_begin += len(camera['object'])
        batched_input = {k: torch.cat(v) if k == 'g_features' else torch.stack(v) for k, v in batched_input.items()}
        batched_input['category_id'] = torch.tensor(category_ids, device=self.device)

        est_pose = self(**batched_input)
        est_pose['orientation_score'] = est_pose['orientation_score']
        obj_idx = 0
        for camera in scene['camera'].values():
            for obj2d in camera['object'].values():
                obj2d.update({k: est_pose[k][obj_idx] for k in keys})
                obj_idx += 1


class ObjectAssociationProposal(pl.LightningModule):
    def __init__(self, sim_scale, **kwargs):
        super().__init__()
        self.save_hyperparameters()

    def affinity_mat(self, embedding1, embedding2):
        embedding1 = embedding1 / (embedding1.norm(dim=-1, keepdim=True) + 1e-8).detach()
        embedding2 = embedding2 / (embedding2.norm(dim=-1, keepdim=True) + 1e-8).detach()
        return torch.sigmoid(self.hparams.sim_scale * (embedding1 @ (embedding2.mT if embedding2.ndim >= 2 else embedding2)))

    def calculate_affinity_for_scene(self, scene):
        if scene.backend is np:
            scene_old, scene = scene, scene.tensor()
        else:
            scene_old = scene

        affinity = {}
        for cam1_id, est_cam1 in scene['camera'].items():
            for cam2_id, est_cam2 in scene['camera'].items():
                if cam1_id == cam2_id or (cam2_id, cam1_id) in affinity:
                    continue
                embedding1 = torch.stack([o['embedding'] for o in est_cam1['object'].values()])
                embedding2 = torch.stack([o['embedding'] for o in est_cam2['object'].values()])
                affinity[(cam1_id, cam2_id)] = self.affinity_mat(embedding1, embedding2)

        if scene_old.backend is np:
            scene_old['affinity'] = Affinity(affinity, backend=scene_old.backend)
        else:
            scene['affinity'] = Affinity(affinity, backend=scene.backend, device=scene.device)


class ObjectProposalLoss(LossModule):
    def __init__(self, train_object_embedding=False, max_affinity_sample=None, func=None, weight=None, **kwargs):
        for k in ('orientation_cls', 'center_depth', 'size_scale', 'offset'):
            assert k in func
            assert k in weight
        assert 'affinity' in weight
        super().__init__(func, weight, **kwargs)
        self.train_object_embedding = train_object_embedding
        self.max_affinity_sample = max_affinity_sample

    def compute_loss(self, est_scene, gt_scene):
        loss = 0.

        # object pose loss
        if not self.train_object_embedding:
            n_object2d = sum(len(c['object']) for c in est_scene['camera'].values())
            for est_camera, gt_camera in zip(est_scene['camera'].values(), gt_scene['camera'].values()):
                for est_obj, gt_obj in zip(est_camera['object'].values(), gt_camera['object'].values()):
                    # direct supervision
                    for key in ('orientation_cls', 'center_depth', 'size_scale', 'offset'):
                        if self.weight[key] is None:
                            continue
                        est_key = 'orientation_score' if key == 'orientation_cls' else key
                        sub_loss = self.func[key](est_obj[est_key], gt_obj[key])
                        self.metrics[f"{key}_loss"].update(sub_loss)
                        if self.weight[key]:
                            loss += self.weight[key] * sub_loss / n_object2d

                    # 3D bounding box corner loss
                    if self.weight['corner'] is not None:
                        est_obj3d = object3d_from_prediction(est_obj, gt_camera)
                        gt_obj3d = object3d_from_prediction(gt_obj, gt_camera)
                        est_bdb3d_corners = bdb3d_corners(est_obj3d)
                        gt_bdb3d_corners = bdb3d_corners(gt_obj3d)
                        corner_loss = self.func['corner'](est_bdb3d_corners, gt_bdb3d_corners)
                        self.metrics['corner_loss'].update(corner_loss)
                    if self.weight['corner']:
                        loss += self.weight['corner'] * corner_loss / n_object2d
        else:
            # affinity loss
            if self.weight['affinity'] is not None:
                # collect all affinity
                est_affinity = torch.cat([a.flatten() for a in est_scene['affinity'].values()])
                gt_affinity = torch.cat([a.flatten() for a in gt_scene['affinity'].values()])
                est_affinity_split = {'pos': est_affinity[gt_affinity], 'neg': est_affinity[~gt_affinity]}

                # limit number of samples to avoid memory overflow
                if self.max_affinity_sample and len(est_affinity) > self.max_affinity_sample:
                    for key, value in est_affinity_split.items():
                        num_sample = max(1, self.max_affinity_sample * len(value) // len(est_affinity))
                        idx = torch.randperm(len(value))[:num_sample]
                        est_affinity_split[key] = value[idx]

                # separately calculate loss for positive and negative affinity for balancing
                affinity_loss_pos = torch.nn.functional.mse_loss(est_affinity_split['pos'], torch.ones_like(est_affinity_split['pos']))
                affinity_loss_neg = torch.nn.functional.mse_loss(est_affinity_split['neg'], torch.zeros_like(est_affinity_split['neg']))
                affinity_loss_pos = 0. if affinity_loss_pos.isnan() else affinity_loss_pos
                affinity_loss_neg = 0. if affinity_loss_neg.isnan() else affinity_loss_neg
                affinity_loss = affinity_loss_pos + affinity_loss_neg
                self.metrics['affinity_loss'].update(affinity_loss)
            if self.weight['affinity']:
                loss += self.weight['affinity'] * affinity_loss

        return loss


class ObjectProposal(PoseProposalModule):
    dataset_cls = ObjectProposalDataset

    def __init__(self, object_pose_proposal, object_association_proposal, input_images=None, loss=None, **kwargs):
        super().__init__(**kwargs)
        self.save_hyperparameters()

        # set input images
        self.config['dataset']['input_images'] = input_images
        object_pose_proposal['input_images'] = input_images

        # initialize object pose proposal network
        if object_pose_proposal['num_category'] == 'auto':
            object_pose_proposal['num_category'] = len(CategoryMapping.object_categories)
        Object2D.load_mean_size(self.config['dataset']['dir'])
        self.object_pose_proposal = ObjectPoseProposalNet(**object_pose_proposal)

        # initialize object association proposal
        self.object_association_proposal = ObjectAssociationProposal(**object_association_proposal)

        # initialize loss
        if loss:
            if 'train_object_embedding' in self.config['train']:
                loss['train_object_embedding'] = self.config['train']['train_object_embedding']
            self.loss = ObjectProposalLoss(**loss)

        # define metrics
        self.val_metrics = nn.ModuleDict({k: torchmetrics.MeanMetric() for k in (
            'orientation', 'center_depth', 'center_depth', 'size', 'offset')})
        self.val_metrics['affinity_AP'] = torchmetrics.classification.BinaryAveragePrecision()
        self.test_metrics = copy.deepcopy(self.val_metrics)

    def configure_optimizers(self):
        params = self.object_pose_proposal.parameters()
        if self.config['train']['train_object_embedding']:
            for param in set(params) - set(self.object_pose_proposal.embedding_mlp.parameters()):
                param.requires_grad_(False)
            params = self.object_pose_proposal.embedding_mlp.parameters()
        params = [{'params': params, 'lr': self.config['train']['lr'], 'name': 'object_pose_proposal'}]
        optimizer = optim.AdamW(params)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, **self.config['train']['lr_scheduler'])
        return [optimizer], [lr_scheduler]

    def forward(self, scene, **kwargs):
        self.object_pose_proposal.calculate_object_pose_for_scene(scene, **kwargs)
        if kwargs.get('get_embedding', True) and (not self.training or self.config['train']['train_object_embedding']):
            self.object_association_proposal.calculate_affinity_for_scene(scene)

    def training_step(self, batch, batch_idx):
        input_data, gt_data = batch
        input_scene = Scene(input_data, backend=torch, device=self.device)
        gt_scene = Scene(gt_data, backend=torch, device=self.device)

        loss = super().train_on_scene(input_scene, gt_scene)

        if input_scene['if_log'] and input_scene.get('affinity'):
            heatmap = []
            for scene in (input_scene.numpy(), gt_scene.numpy()):
                affinity = scene['affinity'][next(iter(scene['affinity'].keys()))]
                heatmap.append(affinity_heatmap(affinity, get_image=True))
            heatmap = image_grid(heatmap, rows=1)
            log_dict = {'global_step': self.global_step, 'affinity_heatmap': wandb.Image(heatmap)}
            self.logger.experiment.log(log_dict)
        return loss

    def eval_batch(self, batch, metrics):
        input_data, gt_data = batch
        input_scene = Scene(input_data, backend=torch, device=self.device)
        self(input_scene)

        for est_camera, gt_camera in zip(input_scene['camera'].values(), gt_data['camera'].values()):
            for est_obj, gt_obj in zip(est_camera['object'].values(), gt_camera['object'].values()):
                for key in ('orientation', 'center_depth', 'size', 'offset'):
                    metric = torchmetrics.functional.mean_absolute_error(est_obj[key], gt_obj[key])
                    if key == 'orientation':
                        metric = torch.rad2deg(torch.min(metric, 2 * np.pi - metric))
                    elif key in ('center_depth', 'size'):
                        metric = metric / gt_obj[key]
                    metrics[key].update(metric)

        if input_scene.get('affinity', False):
            est_affinity = torch.cat([a.flatten() for a in input_scene['affinity'].values()])
            gt_affinity = torch.cat([a.flatten() for a in gt_data['affinity'].values()])
            metrics['affinity_AP'].update(est_affinity, gt_affinity)

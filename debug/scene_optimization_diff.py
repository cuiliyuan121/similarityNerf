from collections import defaultdict
import os
import random
import wandb
import torch
from torch import optim, nn
import torchmetrics
from torchvision import transforms
import numpy as np
from .auto_encoder_nerf import AutoEncoderNeRF
from .base import Front3DDataset, SceneEstimationModule
from utils.dataset import Scene, parameterize_object2d, parameterize_camera,GetFailedError
from utils.torch_utils import LossModule, tensor_linspace
from utils.transform import homotrans, bdb3d_corners, cam2uv, uv2cam, bbox_from_binary_mask, BDB3D_FACES, rotation_mat_dist
from utils.visualize_utils import SceneOptimVisualizer, image_float_to_uint8, image_grid
from utils.general_utils import recursive_getattr
import tempfile
from external.objsdf_general_utils import get_class
from external.pytorch3d_chamfer import directional_chamfer_distance
from tqdm import tqdm
from collections import Counter
from contextlib import suppress
from PIL import Image
from utils.dataset import Scene, Camera
from external.shapenet_renderer_utils import get_archimedean_spiral, look_at
import copy
import yaml
from utils.general_utils import recursive_merge_dict


# @torch.jit.script
def vector_from_pixel_idx(i, j, H, W, focal):
    return torch.stack([(j - W * .5) / focal, -(i - H * .5) / focal, -torch.ones_like(j)], -1)


# @torch.jit.script
def rays_from_pixel(H, W, focal, c2w, i, j):
    dirs = vector_from_pixel_idx(i, j, H, W, focal)
    rays_d = torch.sum(dirs[..., np.newaxis, :].type_as(c2w) * c2w[..., :3, :3], -1)
    viewdirs = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    rays_o = c2w[..., :3, -1].expand(rays_d.shape)
    rays_o, viewdirs = rays_o.reshape(-1, 3), viewdirs.reshape(-1, 3)
    return rays_o, viewdirs


@torch.jit.script
def sample_points_from_rays(ro, vd, near, far, N_samples: int, z_fixed: bool = False):
    # Given ray centre (camera location), we sample z_vals
    # we do not use ray_o here - just number of rays
    if z_fixed:
        z_vals = tensor_linspace(near, far, N_samples)
    else:
        dist = (far - near) / (2 * N_samples)
        z_vals = tensor_linspace(near + dist, far - dist, N_samples)
        near, far = near.unsqueeze(-1), far.unsqueeze(-1)
        z_vals += torch.rand(z_vals.shape, device=ro.device) * (far - near) / (2 * N_samples)
    xyz = ro.unsqueeze(-2) + vd.unsqueeze(-2) * z_vals.unsqueeze(-1)
    vd = vd.unsqueeze(-2).repeat(1, N_samples, 1)
    return xyz, vd, z_vals


@torch.jit.script
def ray_bdb3d_intersection(bdb3d, rays_o, viewdir):
    # https://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-box-intersection
    out_shape = rays_o.shape[:-1]
    rays_o = rays_o.detach().clone().reshape(-1, 3)
    viewdir = viewdir.detach().clone().reshape(-1, 3)

    viewdir[viewdir == 0] = 1.0e-14  # handle divide by zero
    invdir = 1. / viewdir
    neg_sign = (invdir < 0).long()
    pos_sign = 1 - neg_sign

    xmin = (bdb3d[neg_sign[:, 0], 0] - rays_o[:, 0]) * invdir[:, 0]
    xmax = (bdb3d[pos_sign[:, 0], 0] - rays_o[:, 0]) * invdir[:, 0]
    ymin = (bdb3d[neg_sign[:, 1], 1] - rays_o[:, 1]) * invdir[:, 1]
    ymax = (bdb3d[pos_sign[:, 1], 1] - rays_o[:, 1]) * invdir[:, 1]
    zmin = (bdb3d[neg_sign[:, 2], 2] - rays_o[:, 2]) * invdir[:, 2]
    zmax = (bdb3d[pos_sign[:, 2], 2] - rays_o[:, 2]) * invdir[:, 2]

    mask = torch.ones(rays_o.shape[:-1], dtype=torch.bool, device=rays_o.device)
    mask[(xmin > ymax) | (ymin > xmax)] = False
    near_dis = torch.max(xmin, ymin)
    far_dis = torch.min(xmax, ymax)
    mask[(near_dis > zmax) | (zmin > far_dis)] = False
    near_dis = torch.max(near_dis, zmin)
    far_dis = torch.min(far_dis, zmax)

    near_dis[~mask] = float('inf')
    far_dis[~mask] = float('inf')
    near_dis = near_dis.reshape(out_shape)
    far_dis = far_dis.reshape(out_shape)

    return near_dis, far_dis


def render_bdb3d_distance(camera, object3d):
    if isinstance(object3d, list):
        return [render_bdb3d_distance(camera, obj) for obj in object3d]

    j, i = torch.meshgrid(
        torch.linspace(0, camera['width'] - 1, camera['width'], device=camera.device, dtype=camera['cam2world_mat'].dtype),
        torch.linspace(0, camera['height'] - 1, camera['height'], device=camera.device, dtype=camera['cam2world_mat'].dtype), indexing='xy')
    rays_o, viewdir = rays_from_pixel(camera['height'], camera['width'], camera['K'][0, 0], camera['cam2world_mat'], i, j)
    rays_o = homotrans(object3d['world2local_mat'], rays_o)
    viewdir = torch.sum(viewdir[..., None, :] * object3d['rotation_mat'].T, -1)

    bdb3d = torch.stack([-object3d['size'] / 2, object3d['size'] / 2], 0)
    near_dis, far_dis = ray_bdb3d_intersection(bdb3d, rays_o, viewdir)
    near_dis, far_dis = near_dis.reshape(camera['height'], camera['width']), far_dis.reshape(camera['height'], camera['width'])

    return near_dis, far_dis


class SceneOptimizationDataset(Front3DDataset):
    def get_data(self, input_scene, gt_scene):
        if self.config['input_dir'] is None and self.config['noise_std']:
            # add random noise to the GT as input
            input_scene.add_noise_to_3dscene(self.config['noise_std'])

        known_cam_ids = [cam_id for cam_id, cam in input_scene['camera'].items() if 'object' in cam]
        gt_obj_ids = {
            obj_id
            for cam_id in known_cam_ids
            for obj_id in gt_scene['camera'][cam_id].get('object', {}).keys()
        } # dimensionality = 1

        #edit 2023.3.25
        jid = [jid.get('jid') for jid in tuple(gt_scene['object'].values())]
        gt_obj_jids = dict(zip(gt_obj_ids , jid))

        counter_list = Counter(gt_obj_jids.values()).most_common(1)[0][0]
        c = list(filter(lambda k:gt_obj_jids[k] == counter_list, gt_obj_jids))

        gt_obj_single_ids = {
            obj_id
            for obj_id in c
        }
        # ======================================

        for cam_id, camera in gt_scene['camera'].items():
            if 'object' not in camera:
                continue
        
            # generate gt instance_segmap from object segmentation
            gt_instance_segmap = np.ones((camera['height'], camera['width']), dtype=np.int32) * -1
            for obj_idx, (obj_id, object2d) in enumerate(camera['object'].items()):
                if obj_id in gt_obj_single_ids:  #if obj_id in gt_obj_ids: #
                    gt_instance_segmap[object2d['segmentation']] = obj_idx
            camera['image']['instance_segmap'] = gt_instance_segmap

            # generate masked color image
            gt_masked_color = camera['image']['color'].copy()
            gt_masked_color[gt_instance_segmap < 0] = 1
            camera['image']['masked_color'] = gt_masked_color

        # apply transform on image crop
        input_images = self.config['input_images']['nerf_enc_input'] if isinstance(self.config['input_images'], dict) else self.config['input_images']
        if input_images:
            input_scene.crop_object_images()
            for camera in input_scene['camera'].values():
                if 'object' not in camera:
                    continue
                for id, obj2d in camera['object'].items():
                    if id not in gt_obj_single_ids:   # edited 2023.3.25 好像并没有什么用处
                        continue
                    obj2d['camera']['image']['nerf_enc_input'] = obj2d['camera']['image'].preprocess_image(input_images, self.config['resize_width'])

        # set score to 1 for NMS and evaluation
        if 'object' in input_scene:
            for id, obj3d in input_scene['object'].items(): #for obj3d in input_scene['object'].values():
                if 'score' not in obj3d and id in gt_obj_single_ids: # if 'score' not in obj3d
                    obj3d['score'] = 1. #也没有什么用

        # get data as a dict
        input_data = input_scene.data_dict(
            keys=['uid', 'origin_camera_id', 'split'],
            camera={'keys': 'all', 'image': {'keys': ['depth']}, 'object': {'keys':'all', 'camera': {'image': {'keys': ['nerf_enc_input']}}}},
            object={'keys': {'rotation_mat', 'center', 'category_id', 'size', 'category_onehot', 'score', 'latent_code'}}
        ) 

        input_data['object'] = dict((key, value) for key, value in input_data['object'].items() if key in gt_obj_single_ids)

        # one camera
        # input_data['camera'][input_data['origin_camera_id']]['object'] = dict((key, value) for key, value in input_data['camera'][input_data['origin_camera_id']]['object'].items() if key in gt_obj_single_ids) 

        # multi camera
        for id in input_data['camera'].keys():
            input_data['camera'][id]['object'] = dict((key, value) for key, value in input_data['camera'][id]['object'].items() if key in gt_obj_single_ids)

        gt_data = gt_scene.data_dict(
            keys=['uid', 'origin_camera_id', 'split'],
            camera={'image': {'keys': ['color', 'instance_segmap', 'masked_color']}},
            object={'keys': {'rotation_mat', 'center', 'category_id', 'size', 'jid'}}
        )
        
        return input_data, gt_data


class SceneOptimizationLoss(LossModule):
    def __init__(self, proposal_supervision=False, func=None, weight=None, **kwargs):
        for key in ('bdb3d_proj', 'object_size_constraint'):
            assert key in func
        for key in ('gravity', 'bdb3d_proj', 'object_size_constraint', 'chamfer_mask'):
            assert key in weight
        assert not func or 'gravity' not in func and 'chamfer_mask' not in func
        super().__init__(func, weight, **kwargs)
        self.proposal_supervision = proposal_supervision

    def compute_loss(self, est_scene, init_scene):
        loss = 0.

        for object3d in est_scene['object'].values():
            if self.weight['gravity'] is None:
                continue
            gravity_loss = 1 - torch.nn.functional.cosine_similarity(
                object3d['down_vec'],
                torch.tensor([0, 0, -1], device=object3d['down_vec'].device, dtype=object3d['down_vec'].dtype),
                dim=0
            )
            self.metrics['gravity_loss'].update(gravity_loss)
            if self.weight['gravity']:
                loss += self.weight['gravity'] * gravity_loss / len(est_scene['object'])

        n_object2d = sum(len(c['object']) for c in est_scene['camera'].values())
        for camera in est_scene['camera'].values():
            rays = camera['obj_rays']
            valid_rays = rays['hit_obj'] == rays['obj_idx']
            rays = {k: v[valid_rays] for k, v in rays.items() if k != 'hit_obj'}
            for obj_idx, (obj_id, object2d) in enumerate(camera['object'].items()):
                object3d = est_scene['object'][obj_id]

                # bdb3d_proj_loss
                if self.weight['bdb3d_proj'] is not None:
                    corners = bdb3d_corners(object3d)
                    corners = homotrans(camera['world2cam_mat'], corners)
                    corners_2d = cam2uv(camera['K'], corners) - 0.5
                    est_bdb2d = torch.cat([corners_2d.min(dim=0)[0], corners_2d.max(dim=0)[0]])
                    est_bdb2d[[0, 2]].clamp_(0, camera['width'] - 1)
                    est_bdb2d[[1, 3]].clamp_(0, camera['height'] - 1)
                    gt_bdb2d = object2d['bdb2d'].clone()
                    gt_bdb2d[2:] += object2d['bdb2d'][:2]
                    est_bdb2d[:2].clamp_min_(gt_bdb2d[:2])
                    est_bdb2d[2:].clamp_max_(gt_bdb2d[2:])
                    bdb3d_proj_loss = self.func['bdb3d_proj'](est_bdb2d / camera['width'], gt_bdb2d / camera['width'])
                    self.metrics['bdb3d_proj_loss'].update(bdb3d_proj_loss)
                if self.weight['bdb3d_proj']:
                    loss += self.weight['bdb3d_proj'] * bdb3d_proj_loss / n_object2d

                if self.weight['chamfer_mask'] is not None or self.weight['chamfer_depth'] is not None:
                    obj_ray_mask = rays['obj_idx'] == obj_idx
                    obj_rays = {k: v[obj_ray_mask] for k, v in rays.items() if k != 'obj_idx'}

                if self.weight['chamfer_mask'] is not None or self.weight['chamfer_depth'] is not None:
                    # transform hit points from image view to object view
                    with torch.no_grad():
                        est_mask_points = torch.stack([obj_rays['j'], obj_rays['i']], dim=1)
                        hit_points = uv2cam(camera['K'], est_mask_points + 0.5, dis=obj_rays['hit_dis'])
                        hit_points = homotrans(camera['cam2world_mat'], hit_points)
                        hit_points = homotrans(object3d['world2local_mat'], hit_points)
                        hit_points = hit_points / object3d['size']
                    # project hit points back to image view
                    hit_points = hit_points * object3d['size']
                    hit_points = homotrans(object3d['local2world_mat'], hit_points)
                    hit_points = homotrans(camera['world2cam_mat'], hit_points)

                # chamfer_mask_loss
                if self.weight['chamfer_mask'] is not None:
                    gt_mask_points = object2d['segmentation'].nonzero(as_tuple=False).float()
                    gt_mask_points = gt_mask_points + torch.rand_like(gt_mask_points) - 0.5  # jitter
                    est_mask_points = cam2uv(camera['K'], hit_points) - 0.5
                    est_mask_points = torch.flip(est_mask_points, dims=[-1])
                    # calculate 2D chamfer distance between gt_mask_points and est_mask_points
                    chamfer_mask_loss = directional_chamfer_distance(
                        gt_mask_points.unsqueeze(0) / camera['width'],
                        est_mask_points.unsqueeze(0) / camera['width'],
                        norm=1
                    )[0]
                    self.metrics['chamfer_mask_loss'].update(chamfer_mask_loss)
                if self.weight['chamfer_mask']:
                    loss += self.weight['chamfer_mask'] * chamfer_mask_loss / n_object2d

                # chamfer_depth_loss
                if self.weight['chamfer_depth'] is not None:
                    gt_mask_points = object2d['segmentation'].nonzero(as_tuple=False).float()
                    gt_mask_points = torch.flip(gt_mask_points, dims=[-1])
                    gt_mask_depth = camera['image']['depth'][object2d['segmentation']]
                    # gt_mask_depth = gt_mask_depth / gt_mask_depth.mean() * obj_rays['hit_depth'].mean()
                    gt_3d_mask_points = uv2cam(camera['K'], gt_mask_points + 0.5, depth=gt_mask_depth)
                    chamfer_depth_loss = directional_chamfer_distance(
                        gt_3d_mask_points.unsqueeze(0),
                        hit_points.unsqueeze(0),
                        norm=1
                    )[0]
                    self.metrics['chamfer_depth_loss'].update(chamfer_depth_loss)
                if self.weight['chamfer_depth']:
                    loss += self.weight['chamfer_depth'] * chamfer_depth_loss / n_object2d

        # generate object and camera pose parameters
        if self.proposal_supervision:
            parameterize_object2d(est_scene)
            for camera in est_scene['camera'].values():
                parameterize_camera(camera)

        # constraint loss
        for key in self.func.keys():
            key_split = key.split('_')
            if key_split[-1] != 'constraint' or self.weight[key] is None:
                continue
            prefix = key_split[0]
            param = '_'.join(key_split[1:-1])
            if (
                prefix == 'object'  # direct 3D constraint loss
                or prefix == 'camera'
                and self.proposal_supervision is (param in {'pitch', 'roll'})  # camera proposal constraint loss
            ):
                for est_value, init_value in zip(est_scene[prefix].values(), init_scene[prefix].values()):
                    # 3D constraint loss or camera proposal constraint loss
                    constraint_loss = self.func[key](est_value[param], init_value[param])
                    self.metrics[f"{key}_loss"].update(constraint_loss)
                    if self.weight[key]:
                        loss += self.weight[key] * constraint_loss / len(est_scene[prefix])
            elif self.proposal_supervision and prefix == 'object2d':
                # object constraint loss supervised by 2D object proposal
                constraint_loss = 0.
                num_obj = 0
                for est_cam, init_cam in zip(est_scene['camera'].values(), init_scene['camera'].values()):
                    for est_obj, init_obj in zip(est_cam['object'].values(), init_cam['object'].values()):
                        if param == 'orientation':
                            if not isinstance(init_obj[param], torch.Tensor):
                                init_obj[param] = torch.tensor(init_obj[param], device=est_obj[param].device)
                            obj_loss = torch.nn.functional.l1_loss(est_obj[param], init_obj[param])
                            obj_loss = torch.min(obj_loss, 2 * np.pi - obj_loss)
                            obj_loss = self.func[key](obj_loss, torch.tensor(0., device=obj_loss.device))
                        else:
                            obj_loss = self.func[key](est_obj[param], init_obj[param])
                        self.metrics[f"{key}_loss"].update(obj_loss)
                        constraint_loss += obj_loss
                        num_obj += 1
                if self.weight[key]:
                    loss += self.weight[key] * constraint_loss / num_obj

        return loss


class SceneOptimization(SceneEstimationModule):
    dataset_cls = SceneOptimizationDataset

    def __init__(self, autoencoder_nerf, loss, **kwargs):
        super().__init__(**kwargs)
        self.automatic_optimization = False
        self.save_hyperparameters()

        # initialize modules with given hyperparameters
        self.code_embedding = autoencoder_nerf.pop('code_embedding')
        if checkpoint_dir := autoencoder_nerf.pop('checkpoint'):
            autoencoder_nerf.pop('config_dir', None)
            autoencoder_nerf['decoder'].pop('config_dir', None)
            autoencoder_nerf['decoder'].pop('checkpoint', None)
            self.autoencoder_nerf = AutoEncoderNeRF.load_from_checkpoint(checkpoint_dir, **autoencoder_nerf)
        else:
            config_dir = autoencoder_nerf.pop('config_dir')
            print(f"Reading config file from {config_dir} for AutoEncoderNeRF")
            with open(config_dir, 'r') as f:
                config = yaml.load(f, Loader=yaml.FullLoader)
            autoencoder_nerf = recursive_merge_dict(config['model'], autoencoder_nerf)
            if self.code_embedding == 'mean_latent_code':
                autoencoder_nerf['encoder'] = None
            self.autoencoder_nerf = AutoEncoderNeRF(**autoencoder_nerf)

        # initialize loss
        self.loss = SceneOptimizationLoss(**loss)
        self.object_patch_loss = self.autoencoder_nerf.decoder.loss

        # set input images for dataset to be the same as encoder decoder nerf
        self.config['dataset']['input_images'] = self.autoencoder_nerf.config['dataset']['input_images'] if hasattr(self, 'autoencoder_nerf') else None

    def single_object_volume_rendering(self, camera, object3d, rays, batch_size, z_sample,
                                       backward_fcn=None, detach_sample_point=True, white_background=True):
        # generate ray vectors
        rays_o, viewdir = rays_from_pixel(
            camera['height'], camera['width'], camera['K'][0, 0], camera['cam2world_mat'],
            rays['i'], rays['j']
        )

        # sample points from rays
        xyz, viewdir, z_vals = sample_points_from_rays(rays_o, viewdir, rays['sample_near'], rays['sample_far'], z_sample)
        # transform the sampled points and directions to the normalized object coordinate system
        xyz = homotrans(object3d['world2local_mat'], xyz)
        xyz = xyz / object3d['size']
        viewdir = torch.sum(viewdir[..., np.newaxis, :] * object3d['rotation_mat'].T, -1)
        viewdir = viewdir / object3d['size']
        viewdir = viewdir / torch.norm(viewdir, dim=-1, keepdim=True)

        # volume rendering
        if backward_fcn is not None and detach_sample_point:
            # detach rays to avoid gradients from rendering
            xyz, viewdir, z_vals = xyz.detach(), viewdir.detach(), z_vals.detach()
        return self.autoencoder_nerf.decoder.volume_rendering(
            rays_o[None, ...],
            viewdir[None, :, 0, :],
            xyz[None, ...],
            z_vals[None, ...],
            object3d['latent_code'],
            self.config['z_sample'],
            extra_model_outputs=['attention_values'],
            white_background=white_background)

    def render_object_distance(self, camera, object3d_list, num_iter, rays=None, get_distance_map=False):
        if rays is None:
            rays = self.sample_rays_for_scene(camera, object3d_list)

        # render distance map for each object
        valid_rays = rays['exit_dis'] != float('inf')
        obj_dis = torch.ones_like(rays['exit_dis']) * float('inf')  # inf for rays not intersecting with bdb3d
        for obj_idx, object3d in enumerate(object3d_list):
            # render distance for rays intersecting with bdb3d
            valid_obj_rays = valid_rays[:, obj_idx]
            if not valid_obj_rays.any():
                continue

            # generate ray vectors
            rays_o, viewdir = rays_from_pixel(
                camera['height'], camera['width'], camera['K'][0, 0], camera['cam2world_mat'],
                rays['i'][valid_obj_rays], rays['j'][valid_obj_rays]
            )

            # transform rays to the normalized object coordinate system
            rays_o = homotrans(object3d['world2local_mat'], rays_o)
            rays_o = rays_o / object3d['size']
            viewdir = torch.sum(viewdir[..., np.newaxis, :] * object3d['rotation_mat'].T, -1)
            viewdir = viewdir / object3d['size']
            world2local_scale = torch.norm(viewdir, dim=-1)
            viewdir = viewdir / world2local_scale.unsqueeze(-1)
            dis_near, dis_far = [rays[k][valid_obj_rays, obj_idx] * world2local_scale for k in ('entry_dis', 'exit_dis')]

            # render distance with sphere ray marching
            dis = self.autoencoder_nerf.decoder.decoder.sphere_tracing_surface_distance(
                object3d['latent_code'], rays_o, viewdir, dis_init=dis_near, dis_far=dis_far, num_iter=num_iter, extra_model_outputs=['attention_values'])
            obj_dis[valid_obj_rays, obj_idx] = dis / world2local_scale

        if get_distance_map:
            distance_map = torch.ones((camera['height'], camera['width']), dtype=torch.float32, device=camera['K'].device) * float('inf')
            distance_map[rays['i'], rays['j']] = obj_dis.min(dim=-1)[0]
            return distance_map

        rays['obj_dis'] = obj_dis
        return rays

    def sample_rays_for_scene(self, camera, object3d_list, random_ray_sample=None, object2d_list=None,
                              sample_region='bdb3d', max_sqrt_ray_sample=None, min_sample_region_width=None, gt_color=None):
        assert random_ray_sample is None or max_sqrt_ray_sample is None, 'num_ray_sample and max_sqrt_ray_sample cannot be set at the same time'

        # render bdb3d mask for ray sampling
        bdb3d_distances = render_bdb3d_distance(camera, object3d_list)
        bdb3d_distances = [torch.stack(d, dim=-1) for d in zip(*bdb3d_distances)]
        bdb3d_masks = (torch.stack(bdb3d_distances) < float('inf')).any(0).permute(2, 0, 1)

        if sample_region == 'bdb3d':
            assert max_sqrt_ray_sample is None, 'max_sqrt_ray_sample cannot be set when sample_region is bdb3d'
            assert min_sample_region_width is None, 'min_sample_region_width cannot be set when sample_region is bdb3d'
            # sample rays inside each bdb3d mask
            obj_ij = torch.nonzero(bdb3d_masks)
            if random_ray_sample is not None:
                # if num_ray_sample is given, randomly sample rays inside bdb3d mask
                obj_ij = obj_ij[torch.randperm(len(obj_ij))[:random_ray_sample]]
        elif sample_region == 'bdb2d_of_bdb3d':
            assert random_ray_sample is None, 'num_ray_sample cannot be set when sample_region is bdb2d_of_bdb3d'
            # sample rays inside bdb2d of each bdb3d mask
            obj_ij = []
            patch_sizes = []
            for obj_idx, bdb3d_mask in enumerate(bdb3d_masks):
                if not bdb3d_mask.any():
                    patch_sizes.append(None)
                    continue
                cmin, rmin, w, h = bbox_from_binary_mask(bdb3d_mask)
                # calculate downsample stride to ensure w * h is less than max_sqrt_ray_sample ** 2
                if max_sqrt_ray_sample is not None:
                    downsample = int(np.ceil(np.sqrt(w * h / max_sqrt_ray_sample ** 2)))
                    w, h = w // downsample, h // downsample
                else:
                    downsample = 1
                # skip if the patch is too small
                if w < min_sample_region_width or h < min_sample_region_width:
                    patch_sizes.append(None)
                    continue
                patch_sizes.append((w, h))
                c = torch.linspace(cmin, cmin + (w - 1) * downsample, w, dtype=torch.int64, device=camera['K'].device)
                r = torch.linspace(rmin, rmin + (h - 1) * downsample, h, dtype=torch.int64, device=camera['K'].device)
                # randomly shift c and r to avoid sampling rays on the same pixel
                if downsample > 1:
                    c += torch.randint(0, downsample, (1, ), device=c.device)
                    r += torch.randint(0, downsample, (1, ), device=r.device)
                j, i = torch.meshgrid(c, r, indexing='xy')
                i, j = i.flatten(), j.flatten()
                obj_ij.append(torch.stack([torch.full_like(i, obj_idx, dtype=torch.int64), i, j], dim=-1))
            obj_ij = torch.cat(obj_ij, dim=0) if len(obj_ij) > 0 else torch.empty((0, 3), dtype=torch.int64, device=camera['K'].device)
        else:
            raise ValueError(f"Unknown sample_region: {sample_region}")

        rays = {'obj_idx': obj_ij[..., 0], 'i': obj_ij[..., 1], 'j': obj_ij[..., 2]}
        rays['entry_dis'], rays['exit_dis'] = [d[rays['i'], rays['j']] for d in bdb3d_distances]
        # if camera is inside a bdb3d, it is possible that near > far
        rays['entry_dis'][rays['entry_dis'] > rays['exit_dis']] = 0

        # initialize sample range with entry/exit distance
        obj_entry_dis = rays['entry_dis'][range(len(rays['obj_idx'])), rays['obj_idx']]
        obj_exit_dis = rays['exit_dis'][range(len(rays['obj_idx'])), rays['obj_idx']]
        # if rays does not hit bdb3d, entry/exit distance should be the min/max distance of the bdb3d
        rays_hitting_bdb3d = bdb3d_masks[rays['obj_idx'], rays['i'], rays['j']]
        if not rays_hitting_bdb3d.all():
            obj_entry_dis[~rays_hitting_bdb3d] = obj_entry_dis[rays_hitting_bdb3d].min()
            obj_exit_dis[~rays_hitting_bdb3d] = obj_exit_dis[rays_hitting_bdb3d].max()
        # near far range should be larger than the bdb3d intersection range
        rays['sample_near'] = obj_entry_dis.clone() - self.config['surface_near']
        rays['sample_far'] = obj_exit_dis.clone() + self.config['surface_far']

        # find the hit object, distance, gt for each ray
        with torch.no_grad():
            self.render_object_distance(camera, object3d_list, self.config['sphere_tracing_iter'], rays=rays)

        # find the hit object, distance for each ray
        rays['hit_dis'], rays['hit_obj'] = torch.min(rays['obj_dis'], dim=-1)
        # if a ray does not hit any object, hit_obj = -1
        rays['hit_obj'][rays['hit_dis'] == float('inf')] = -1

        hit_target_obj = rays['hit_obj'] == rays['obj_idx']

        if self.config['safe_region_render']:
            # for rays hitting the target object, sample points near the surface
            rays['sample_near'][hit_target_obj] = rays['hit_dis'][hit_target_obj] - self.config['surface_near']
            rays['sample_far'][hit_target_obj] = rays['hit_dis'][hit_target_obj] + self.config['surface_far']

        # for rays entering the target bdb3d before hitting another object, sample points from entry_dis
        hit_another_obj = ~hit_target_obj & (obj_entry_dis < rays['hit_dis'])
        rays['sample_far'][hit_another_obj] = torch.min(obj_exit_dis[hit_another_obj], rays['hit_dis'][hit_another_obj])

        # for rays occluded by another object (hit another object before entering target bdb3d), no supervision should be applied
        occluded_by_another_obj = ~hit_target_obj & (obj_entry_dis > rays['hit_dis'])
        if sample_region == 'bdb2d_of_bdb3d' and random_ray_sample is None:
            rays['occluded'] = ~hit_target_obj & (obj_entry_dis > rays['hit_dis'])
        else:
            rays = {k: v[~occluded_by_another_obj] for k, v in rays.items()}

        if object2d_list is not None and gt_color is not None:
            # get gt_seg from object segmentation
            obj_masks = torch.stack([o['segmentation'] for o in object2d_list], dim=0)
            rays['gt_seg'] = obj_masks[rays['obj_idx'], rays['i'], rays['j']]

            # get gt_rgb from color image and object segmentation
            rays['gt_rgb'] = gt_color[rays['i'], rays['j']]
            rays['gt_rgb'][~rays['gt_seg']] = 1

        # rays should not propagate gradient to pose
        rays = {k: v.detach() for k, v in rays.items()}

        if sample_region == 'bdb2d_of_bdb3d' and random_ray_sample is None:
            return rays, patch_sizes
        return rays

    def object_novel_view_synthesis(self, camera, object3d_list, batch_size, z_sample,
                                    rays=None, backward_fcn=None, get_img=False, patch_sizes=None, detach_sample_point=True):
        if rays is None:
            rays = self.sample_rays_for_scene(camera, object3d_list)

        # disable batched rendering for image patches
        do_patch_optim = backward_fcn is not None and patch_sizes is not None
        if do_patch_optim:
            batch_size = None

        # initialize results
        est_rays = {
            'est_rgb': torch.ones((len(rays['obj_idx']), 3), dtype=torch.float32, device=camera['K'].device),
            'est_seg': torch.zeros(len(rays['obj_idx']), dtype=torch.bool, device=camera['K'].device),
            'est_alpha': torch.zeros((len(rays['obj_idx']), 1), dtype=torch.float32, device=camera['K'].device),
        }

        # render each object
        for obj_idx, object3d in enumerate(object3d_list):
            # get rays for the current object
            obj_ray_mask = rays['obj_idx'] == obj_idx
            if obj_ray_mask.sum() == 0:
                continue
            obj_rays = {k: v[obj_ray_mask] for k, v in rays.items() if k != 'obj_idx'}

            # render the current object
            obj_results = self.single_object_volume_rendering(
                camera, object3d, obj_rays, batch_size, z_sample,
                (lambda l: backward_fcn(l, retain_graph=True)) if backward_fcn is not None else None,
                detach_sample_point=detach_sample_point, white_background=do_patch_optim,
            )

            # update results
            est_rays['est_rgb'][obj_ray_mask] = obj_results['rgb'].detach()
            est_rays['est_seg'][obj_ray_mask] = obj_results['min_sdf'] < 0
            est_rays['est_alpha'][obj_ray_mask] = obj_results['mask'].detach()

            if do_patch_optim:
                # avoid occluded rays
                patch_size = patch_sizes[obj_idx]
                est_rgb = obj_results['rgb'].clone()
                gt_rgb = rays['gt_rgb'][obj_ray_mask]
                est_rgb[obj_rays['occluded']] = gt_rgb[obj_rays['occluded']]

                # compute object patch loss
                est = {
                    'rgb': est_rgb.reshape(patch_size[1], patch_size[0], 3),
                    'min_sdf': obj_results['min_sdf'][~obj_rays['occluded']]
                }
                gt = {
                    'rgb': gt_rgb.reshape(patch_size[1], patch_size[0], 3),
                    'seg': rays['gt_seg'][obj_ray_mask][~obj_rays['occluded']]
                }
                object_patch_loss = self.object_patch_loss(est, gt)
                if object_patch_loss:
                    backward_fcn(object_patch_loss, retain_graph=True)

        results = {}
        if backward_fcn is not None:
            results['loss'] = self.object_patch_loss.compute_metrics()

        if get_img:
            # remove occluded rays for visualization
            if 'occluded' in rays:
                rays = {k: v[~rays['occluded']] for k, v in rays.items() if k != 'occluded'}

            # ignore rays that does not hit target object
            valid_rays = rays['hit_obj'] == rays['obj_idx']

            # draw color image on a white background from valid rays
            results['color'] = torch.ones((camera['height'], camera['width'], 3), dtype=torch.float32, device=camera['K'].device)
            results['color'][rays['i'][valid_rays], rays['j'][valid_rays]] = est_rays['est_rgb'][valid_rays]

            results['alpha'] = torch.zeros((camera['height'], camera['width'], 1), dtype=torch.float32, device=camera['K'].device)
            results['alpha'][rays['i'][valid_rays], rays['j'][valid_rays]] = est_rays['est_alpha'][valid_rays]
            # draw instance segmentation from all rays
            if self.config['segmap_foreground_thres']:
                instance_segmap = torch.zeros((camera['height'], camera['width'], len(object3d_list)), dtype=torch.bool, device=camera['K'].device)
                instance_segmap[rays['i'], rays['j'], rays['obj_idx']] = est_rays['est_seg']
                instance_segmap = torch.cat([torch.ones_like(instance_segmap[:, :, :1]) * self.config['segmap_foreground_thres'], instance_segmap], dim=-1)
                results['instance_segmap'] = instance_segmap.argmax(dim=-1) - 1

            # draw ray hit map from valid rays
            results['instance_hitmap'] = torch.ones((camera['height'], camera['width']), dtype=torch.int64, device=camera['K'].device) * -1
            results['instance_hitmap'][rays['i'][valid_rays], rays['j'][valid_rays]] = rays['obj_idx'][valid_rays]

        return results

    def init_latent_code(self, est_scene, overwrite=True):
        if self.code_embedding == 'encoder':
            est_scene.aggregate_object_images()
            # embed latent code with encoder
            for object3d in est_scene['object'].values():
                if not overwrite and 'latent_code' in object3d:
                    continue
                object3d['latent_code'] = self.autoencoder_nerf.embed_latent_code(object3d)
        elif self.code_embedding == 'mean_latent_code':
            # initialize latent code with mean values
            for object3d in est_scene['object'].values():
                if not overwrite and 'latent_code' in object3d:
                    continue
                object3d['latent_code'] = self.autoencoder_nerf.decoder.mean_latent_code(object3d['category_id'])
                break
            self.delivery_address(est_scene)

        else:
            raise ValueError(f"Unknown code embedding method: {self.code_embedding}")

    def render_scene(self, est_scene, camera_dict=None):
        if camera_dict is None:
            for camera in est_scene['camera'].values():
                results = self.object_novel_view_synthesis(
                    camera, list(est_scene['object'].values()), self.config['batch_size'], self.config['z_sample'], get_img=True)
                camera['image']['color'] = results['color']
                camera['image']['alpha'] = results['alpha']
        else:
            new_camera_dict = {}
            for cam_id, camera in camera_dict.items():
                results = self.object_novel_view_synthesis(
                    camera, list(est_scene['object'].values()), self.config['batch_size'], self.config['z_sample'], get_img=True)
                new_camera_dict[cam_id] = {'image': {'color': results['color'], 'alpha': results['alpha']}}
            return new_camera_dict

    def update_latent_code(self, scene):
        for object3d in scene['object'].values():
            object3d['latent_code'] = self.autoencoder_nerf.embed_latent_code(object3d)

    def delivery_address(self, scene):
        if not 'object' in scene:
            return
        ids = list(scene['object'].keys()) #
        for id_ in ids[1:]:
            scene['object'][id_]['latent_code'] = scene['object'][ids[0]]['latent_code']
    
    def print_address(self, est_scene):
        print("est_scene:")
        for temo_obj in est_scene['object'].values():
            print(id(temo_obj['latent_code']))

    def optimize_scene(self, est_scene, gt_scene, skip_stage=None):
        wandb_logger = self.logger.experiment
        if skip_stage is None:
            skip_stage = self.config['test']['skip_stage']
        if isinstance(skip_stage, str):
            skip_stage = [skip_stage]

        # log ground truth scene vis
        if est_scene['if_log']:
            scene_optim_vis = SceneOptimVisualizer(gt_scene.numpy(), **self.config['test']['scene_optim_vis'])

        def visualize_frame(scene):
            if not est_scene['if_log']:
                return
            with torch.no_grad():
                scene = scene.clone()
                if 'video' in scene_optim_vis.vis_types:
                    for camera_id, camera in scene['camera'].items():
                        results = self.object_novel_view_synthesis(
                            camera, list(scene['object'].values()), self.config['batch_size'], self.config['z_sample'], get_img=True)
                        camera['image'].update(results)
                scene_optim_vis.add_frame(scene.numpy())

        # Optimize scene
        with torch.enable_grad(), torch.inference_mode(False):
            
            # clone the scene for optimization
            est_scene = est_scene.clone() #新的tensor开辟新的内存，但是仍然留在计算图中。 torch.clone () 操作在不共享数据内存的同时支持梯度回溯，所以常用在神经网络中某个单元需要重复使用的场景下
            
            self.delivery_address(est_scene)
            self.print_address(est_scene)

            # visualize the initial scene
            if self.code_embedding == 'embed_image_feature':
                self.update_latent_code(est_scene)
            init_scene = est_scene.clone(now=True)
            self.delivery_address(init_scene)#
            visualize_frame(init_scene)
    
            # optimization schedule
            optim_schedule = self.config['test']['optim_schedule']
            num_opts = self.config['test']['num_opts']
            if optim_schedule:
                def compute_steps(schedule, start=0, end=num_opts):
                    steps = [int(stage['proportion'] * (end - start)) for stage in schedule]
                    end_steps = np.cumsum(steps) + start
                    end_steps[-1] = end
                    start_steps = np.concatenate([[start], end_steps[:-1]])
                    for stage, start_step, end_step in zip(schedule, start_steps, end_steps):
                        stage['start_step'] = start_step
                        stage['end_step'] = end_step

                # expand cycles
                compute_steps(optim_schedule)
                expanded_schedule = []
                for stage in optim_schedule:
                    if 'cycle' in stage:
                        cycle_steps = np.linspace(stage['start_step'], stage['end_step'], stage['cycle'] + 1).astype(np.int)
                        cycle_start_steps = cycle_steps[:-1]
                        cycle_end_steps = cycle_steps[1:]
                        for c, cycle_start_step, cycle_end_step in zip(range(stage['cycle']), cycle_start_steps, cycle_end_steps):
                            cycle_schedule = copy.deepcopy(stage['schedule'])
                            for s in cycle_schedule:
                                s['name'] = f"{s['name']}-{stage['name']}-cycle_{c}"
                            compute_steps(cycle_schedule, cycle_start_step, cycle_end_step)
                            expanded_schedule.extend(cycle_schedule)
                    else:
                        expanded_schedule.append(stage)
                optim_schedule = expanded_schedule
            else:
                optim_schedule = [{'name': 'default', 'start_step': 0, 'end_step': num_opts}]

            # start optimization and log the loss and image output
            i = 0
            with tqdm(total=num_opts, desc='Optimizing scene', leave=False) as pbar:
                while i < num_opts:
                    # initialize optimization at the start of each stage
                    stage = None
                    for stage in optim_schedule:
                        if i >= stage['start_step'] and i < stage['end_step']:
                            break
                    assert stage is not None

                    if i == stage['start_step']:
                        # configure optimizer
                        lr = self.config['test']['lr'].copy()
                        update_lr = stage.get('lr', {}).copy()
                        for key, value in lr.items():
                            if value == 'override_with_null':
                                lr[key] = None
                                update_lr.pop(key, None)
                        lr.update(update_lr)

                        # skip stage if specified
                        if skip_stage and stage['name'] in skip_stage:
                            tqdm.write(f"Skipping stage: {stage['name']}, "
                                       f"step: [{stage['start_step']}, {stage['end_step']}), "
                                       f"exp: {self.logger.experiment.config['args']['id']}")
                            skip_opts = stage['end_step'] - stage['start_step']
                            i += skip_opts
                            pbar.update(skip_opts)
                            continue
                        else:
                            tqdm.write(f"Optimizing scene {est_scene['uid']} with stage: {stage['name']}, "
                                       f"step: [{stage['start_step']}, {stage['end_step']})")

                        params = est_scene.enable_optimization(lr) 
                        if self.code_embedding == 'embed_image_feature':
                            self.update_latent_code(est_scene)
                        opt = get_class(self.config['test']['optimizer'])(params)
                        if 'lr_scheduler' in stage:
                            lr_scheduler = optim.lr_scheduler.StepLR(opt, **stage['lr_scheduler'])

                        # configure loss weights
                        for loss, weight in stage.get('loss_weight', {}).items():
                            recursive_getattr(self, loss).update_weight_based_on_init(weight)

                    opt.zero_grad()
                    log_dict = defaultdict(list)

                    # compute and log per camera loss
                    random_ray_sample = self.config['test']['random_ray_sample']
                    sample_region = self.config['test']['sample_region']
                    if self.object_patch_loss.is_enabled():
                        assert sample_region == 'bdb2d_of_bdb3d', 'sample_region must be bdb2d_of_bdb3d when using object_patch_loss'
                    for camera_id, camera in est_scene['camera'].items():
                        # if not len(camera['object'] ):
                        #     continue 
                        obj3d_list = [est_scene['object'][k] for k in camera['object'].keys()]
                        obj_rays = self.sample_rays_for_scene(
                            camera,
                            obj3d_list,
                            random_ray_sample=random_ray_sample // len(est_scene['camera']) if random_ray_sample else None,
                            object2d_list=list(camera['object'].values()),
                            sample_region=sample_region,
                            max_sqrt_ray_sample=self.config['test']['max_sqrt_ray_sample'],
                            min_sample_region_width=self.config['test']['min_sample_region_width'],
                            gt_color=gt_scene['camera'][camera_id]['image']['color'],
                        )
                        obj_rays, patch_sizes = obj_rays if isinstance(obj_rays, tuple) else (obj_rays, None)

                        # render and backpropagate gradients
                        results = self.object_novel_view_synthesis(
                            camera, obj3d_list, self.config['batch_size'], self.config['z_sample'],
                            obj_rays, self.manual_backward, patch_sizes=patch_sizes, detach_sample_point=self.config['test']['detach_sample_point'])
                        for k, v in results['loss'].items():
                            log_dict[k].append(v)
                        camera['obj_rays'] = obj_rays
                    for k, v in log_dict.items():
                        log_dict[k] = np.mean(v)

                    # compute and log other losses
                    loss = self.loss(est_scene, init_scene)
                    if loss:
                        self.manual_backward(loss, retain_graph=True)
                    log_dict.update(self.loss.compute_metrics())

                    # step and update attributes referencing the optimized parameters
                    opt.step()# change latent code

                    if 'lr_scheduler' in stage:
                        lr_scheduler.step()
                    est_scene.step()
                    if self.code_embedding == 'embed_image_feature':
                        self.update_latent_code(est_scene)

                    # visualize the optimization process
                    visualize_frame(est_scene)

                    # rename log_dict keys and log to wandb
                    log_dict = {f"test/optimization/{k}": v for k, v in log_dict.items()}
                    log_dict['global_step'] = est_scene['batch_idx'] * num_opts + i
                    wandb_logger.log(log_dict)

                    i += 1
                    pbar.update(1)

        # align object poses with vertical axis
        for obj3d in est_scene['object'].values():
            forward_vec = obj3d['rotation_mat'][:, 1].clone()
            forward_vec[2] = 0.
            forward_vec = forward_vec / torch.linalg.norm(forward_vec)
            up_vec = torch.tensor([0., 0., 1.], device=forward_vec.device)
            right_vec = torch.cross(forward_vec, up_vec)
            obj3d['rotation_mat'] = torch.stack([right_vec, forward_vec, up_vec], dim=1)
        visualize_frame(est_scene)

        # log optimization process to wandb
        if est_scene['if_log']:
            log_dict = {"global_step": est_scene['batch_idx']}
            if 'wireframe' in scene_optim_vis.vis_types:
                # log scene optimization as plotly and save as html
                # https://github.com/wandb/wandb/issues/2014
                scene_optim_fig = scene_optim_vis.scene_optim_vis()
                with tempfile.NamedTemporaryFile(mode='r+', suffix='.html') as f:
                    scene_optim_fig.write_html(f, auto_play=False)
                    log_dict["test/optimization/scene_optim_vis"] = wandb.Html(f)
            if 'video' in scene_optim_vis.vis_types:
                fps = min(len(scene_optim_vis.video_frames) / self.config['test']['min_video_length'], 24)
                log_dict["test/optimization/generated"] = wandb.Video(
                    np.stack(scene_optim_vis.video_frames), fps=fps, format="mp4", caption=gt_scene['uid'])
            wandb_logger.log(log_dict)

        return est_scene

    # ======================= edit 2023.3.36
    def render_novel(self, est_scene, camera_pose, folder):
        empty_camera = next(iter(est_scene['camera'].values())).copy()

        # render novel views
        camera_dict = {} # new camera 
        for cam_id, camera_pose in enumerate(camera_pose):
            camera = empty_camera.copy()
            camera['cam2world_mat'] = camera_pose
            camera_dict[cam_id] = camera #根据每个新的pose对应camera
        
        camera_dict = self.render_scene(est_scene, camera_dict) #render novel camera 

        #save generated novel views
        wandb_logger = self.logger.experiment
        output_scene_root = os.path.join(wandb_logger.config['args']['output_dir'], wandb_logger.config['args']['id'], 'scene')
        output_scene_dir = os.path.join(output_scene_root, est_scene['uid'])
        output_frame_dir = os.path.join(output_scene_dir, folder)
        os.makedirs(output_frame_dir, exist_ok=True)
        for cam_id, cam in camera_dict.items():
            frame = image_float_to_uint8(Camera(cam)['image']['color'])
            Image.fromarray(frame).save(os.path.join(output_frame_dir, f"{cam_id:04d}.jpg"))
    
    def test_step(self, batch, batch_idx):
        input_data, gt_data = batch
        est_scene, gt_scene = Scene(input_data, backend=torch, device=self.device), Scene(gt_data, backend=torch, device=self.device)

        num_optim = 3
        est_scene_original = est_scene.copy()
        gt_scene_original = gt_scene.copy()

        cam_id = set(random.sample(set(est_scene['camera'].keys()), num_optim))
        if est_scene['origin_camera_id'] not in cam_id:
            cam_id.pop()
        cam_id.add(est_scene['origin_camera_id'])
       
        est_scene = est_scene.subscene(cam_id)
        gt_scene = gt_scene.subscene(cam_id)
        est_scene_novel =  est_scene_original.subscene(set(est_scene_original['camera'].keys()) - cam_id)
        gt_scene_novel =  gt_scene_original.subscene(set(gt_scene_original['camera'].keys()) - cam_id)

        if not self.config['test']['skip_prediction']:
            # Embed latent codes for all objects
            self.init_latent_code(est_scene)

            # Then optimize scene
            est_scene = self.optimize_scene(est_scene, gt_scene)

            # render final scene into color images
            self.render_scene(est_scene)

            # render est_novel_scene with opted latent_code
            ids1 = list(est_scene_novel['object'].keys())
            ids2 = list(est_scene['object'].keys())
            for i in ids1:
                if i not in ids2: #est_scene  中没有的场景
                    continue 
                est_scene_novel['object'][i]['latent_code'] = est_scene['object'][i]['latent_code']
            self.render_scene(est_scene_novel)

            # save estimated scene
            self.save_scene_result(est_scene)
            self.save_scene_result(est_scene_novel)
            
        # Finally evaluate the scene
        self.eval_scene(est_scene, gt_scene, est_scene_novel, gt_scene_novel)  # self.eval_scene(est_scene, gt_scene)
        self.print_address(est_scene)

        # visualize scene
        self.visualize_scene(est_scene, gt_scene)

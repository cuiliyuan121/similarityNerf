from collections import defaultdict
import wandb
import torch
import numpy as np
import torch.nn.functional as F
import tempfile
from external.nerf_from_image.models.generator import Generator
from utils.visualize_utils import image_grid, image_float_to_uint8
import matplotlib.pyplot as plt
import os
from external.nerf_from_image.lib import pose_estimation, nerf_utils, pose_utils
import torchmetrics
import lpips
from external.nerf_from_image import augment
from collections import defaultdict
import json
from torch import optim, nn
from torch.utils.data import Dataset
import pytorch_lightning as pl
from utils.dataset import ObjectNeRF, CategoryMapping
from utils.torch_utils import MyLightningModule, LossModule, initialize_cls


class StyleNeRFDataset(Dataset):
    def __init__(self, config, split):
        self.config = config
        self.split = split
        self.data_dict_keys = ['jid', 'category', 'category_id', 'camera']

        # load split
        self.object_ids, self.object_categories = self.load_split(self.config['dir'])
        print(f"Got {len(self.object_ids)} objects for {split} split")

        # default not to crop the image
        self.crop_img = False

        # load category mapping
        CategoryMapping.load_category_mapping(self.config['dir'])

    def load_split(self, dir):
        split_dir = os.path.join(dir, self.split + '.json')
        with open(split_dir) as f:
            object_id_categories = json.load(f)
        assert isinstance(object_id_categories[0], list) and len(object_id_categories[0]) == 2, \
            "Split file should be a list of [object_id, category], you probably forgot to run filter object job."
        if self.config['category'] != 'all':
            print(f"Filtering dataset by category: {self.config['category']}")
            object_ids = [obj for obj in object_id_categories if obj[1] == self.config['category']]
            assert len(object_ids) > 0, f"No objects found for category {self.config['category']}"
        else:
            object_ids = [obj for obj in object_id_categories]
        object_ids.sort(key=lambda x: x[0])
        return zip(*object_ids)

    def __len__(self):
        return len(self.object_ids)

    def load_and_sample_object(self, object_dir, idx):
        objnerf = ObjectNeRF.from_dir(object_dir)
        if self.split == 'train':
            objnerf = objnerf.random_subset(self.config['train_camera_num'])
        else:
            stride_from_camera = self.config['stride_from_camera']
            if stride_from_camera == 'rotate':
                stride_from_camera = (idx % self.config['eval_camera_stride']) % len(objnerf['camera'])
            objnerf = objnerf.subset(list(objnerf['camera'].keys())[stride_from_camera::self.config['eval_camera_stride']], relabel_camera=False)
        return objnerf

    def __getitem__(self, idx):
        # load object
        object_dir = os.path.join(self.config['dir'], self.object_ids[idx])
        objnerf = self.load_and_sample_object(object_dir, idx)

        # crop image if needed
        for camera in objnerf['camera'].values():
            img = camera['image']['color']
            if self.crop_img:
                img = img[32:-32, 32:-32]
                camera['height'], camera['width'] = img.shape[:2]
                camera['K'][:2, -1] -= 32

        # get data as a dict
        data = objnerf.data_dict(self.data_dict_keys)
        data['idx'] = torch.tensor(idx)
        data['category_id'] = torch.tensor(data['category_id'])

        return data


class StyleNeRFLoss(LossModule):
    def __init__(self, lpips_net, lpips_aug, func, weight=None, **kwargs):
        assert 'rgb' in func
        assert 'lpips' in weight
        if not weight or 'rgb' not in weight:
            weight['rgb_weight'] = 1.
        super().__init__(func, weight, **kwargs)
        self.lpips_loss = lpips.LPIPS(net=lpips_net)
        self.lpips_aug = lpips_aug
        self.metrics['psnr'] = torchmetrics.PeakSignalNoiseRatio(data_range=1.0)

    def compute_loss(self, est, gt):
        loss = 0.

        if self.weight['lpips'] is not None:
            est_img_lpips = est['rgb'].permute(2, 0, 1).unsqueeze(0)
            gt_img_lpips = gt['rgb'].permute(2, 0, 1).unsqueeze(0)
            if self.lpips_aug:
                est_gt_img = torch.cat((est_img_lpips, gt_img_lpips), dim=1)
                est_gt_img = est_gt_img.unsqueeze(1).expand(-1, 15, -1, -1, -1).contiguous().flatten(0, 1)
                est_gt_img, _, _ = augment.augment(est_gt_img, None, None, 1.0)
                est_img_lpips = torch.cat((est_img_lpips, est_gt_img[:, :3]), dim=0)
                gt_img_lpips = torch.cat((gt_img_lpips, est_gt_img[:, 3:]), dim=0)
            lpips_loss = self.lpips_loss(est_img_lpips, gt_img_lpips, normalize=True).mean()
            self.metrics['lpips_loss'].update(lpips_loss)
        if self.weight['lpips']:
            loss += lpips_loss * self.weight['lpips']

        if self.weight['rgb'] is not None:
            rgb_loss = self.func['rgb'](est['rgb'], gt['rgb'].clone())
            self.metrics['rgb_loss'].update(rgb_loss, weight=len(gt['rgb']))
            self.metrics['psnr'].update(est['rgb'], gt['rgb'])
        if self.weight['rgb']:
            loss += self.weight['rgb'] * rgb_loss

        if 'seg' in gt:
            if self.weight.get('seg', None) is not None:
                wrong_sdf = torch.cat([
                    est['min_sdf'][gt['seg'] & (est['min_sdf'] > 0)],
                    -est['min_sdf'][~gt['seg'] & (est['min_sdf'] < 0)]
                ])
                seg_loss = torch.mean(wrong_sdf) if len(wrong_sdf) > 0 else 0.
                self.metrics['seg_loss'].update(seg_loss, weight=len(gt['seg']))
            if self.weight.get('seg', None):
                loss += self.weight['seg'] * seg_loss

        return loss


def sample_rays(height, width, focal_length, tform_cam2world, bbox=None, center=None, scene_range=1):

    ray_origins, ray_directions = nerf_utils.get_ray_bundle(
        height, width, focal_length, tform_cam2world, bbox, center)

    ray_directions = F.normalize(ray_directions, dim=-1)
    with torch.no_grad():
        near_thresh, far_thresh = nerf_utils.compute_near_far_planes(
            ray_origins.detach(), ray_directions.detach(), scene_range)
    ray_origins = ray_origins.flatten(1, 2) # bs*n_rays*3
    ray_directions = ray_directions.flatten(1, 2)
    near_thresh = near_thresh.flatten(1, 2)
    far_thresh = far_thresh.flatten(1, 2)
    return ray_origins, ray_directions, near_thresh, far_thresh


def sample_points_from_rays(ray_origins,
                            ray_directions,
                            near_thresh,
                            far_thresh,
                            depth_samples_per_ray,
                            randomize=True):
    query_points, depth_values = nerf_utils.compute_query_points_from_rays(
        ray_origins,
        ray_directions,
        near_thresh,
        far_thresh,
        depth_samples_per_ray,
        randomize=randomize,
    )
    return query_points, depth_values


def fine_sample_rays(ray_origins, ray_directions, depth_values,
                     sigma,
                     depth_samples_per_ray=64,
                     randomize=True):
    z_vals = depth_values
    with torch.no_grad():
        weights = nerf_utils.render_volume_density_weights_only(
            sigma.squeeze(-1), ray_origins, ray_directions,
            depth_values).flatten(0, -2)

        # Smooth weights as in EG3D
        weights = F.max_pool1d(weights.unsqueeze(1).float(), 2, 1, padding=1)
        weights = F.avg_pool1d(weights, 2, 1).squeeze(-2)
        weights = weights + 0.01

        z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        z_samples = nerf_utils.sample_pdf(
            z_vals_mid.flatten(0, -2),
            weights[..., 1:-1],
            depth_samples_per_ray,
            deterministic=not randomize,
        )
        z_samples = z_samples.view(*z_vals.shape[:-1], z_samples.shape[-1])

    z_values_sorted, z_indices_sorted = torch.sort(torch.cat((z_vals, z_samples), dim=-1), dim=-1)
    query_points_fine = ray_origins[..., None, :] + ray_directions[..., None, :] * z_samples[..., :, None]

    return query_points_fine, z_indices_sorted, z_values_sorted


class StyleNeRF(MyLightningModule):
    dataset_cls = StyleNeRFDataset

    def __init__(self, decoder, loss, num_object=None, **kwargs):
        super().__init__(**kwargs)
        self.automatic_optimization = False
        if num_object == 'auto':
            num_object = len(self.dataset('train'))
        self.save_hyperparameters()

        # initialize modules with given hyperparameters
        self.decoder = Generator(**decoder)
        self.loss = StyleNeRFLoss(**loss)

        # initialize latent code GT
        if num_object:
            self.latent_codes = nn.Embedding(num_object, decoder['latent_dim'])

        # properly define torchmetrics to allow automatic device moving
        # https://torchmetrics.readthedocs.io/en/stable/pages/overview.html?highlight=device#metrics-and-devices
        CategoryMapping.load_category_mapping(self.config['dataset']['dir'])
        self.test_metrics = nn.ModuleDict({
            c: nn.ModuleDict({
                k: torchmetrics.MeanMetric() for k in ('psnr', 'ssim')
            }) for c in CategoryMapping.object_categories + ['mean']
        })
        self.val_metrics = nn.ModuleDict({k: torchmetrics.MeanMetric() for k in ('psnr', 'ssim')})

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, **kwargs):
        print("Checking checkpoint compatibility")
        checkpoint = torch.load(checkpoint_path)
        if 'state_dict' not in checkpoint:
            print("Checkpoint is not from pytorch-lightning, using compatible loading method")
            model = initialize_cls(cls, **kwargs)
            model.decoder.load_state_dict(checkpoint['model_ema'])

            # init latent code
            if hasattr(model, 'latent_codes'):
                dataset = model.dataset('train')
                category_ids = [CategoryMapping.object_categories.index(c) for c in dataset.object_categories]
                category_ids = torch.Tensor(category_ids, device=model.device)
                latent_codes = nn.Parameter(torch.Tensor(model.latent_codes.weight.shape)).requires_grad_(False)
                for i in range(len(CategoryMapping.object_categories)):
                    if i not in category_ids:
                        continue
                    latent_codes[category_ids == i] = model.mean_latent_code(i)[0, 0]
                model.latent_codes.weight = latent_codes.requires_grad_()

            return model
        else:
            print("Checkpoint is from pytorch-lightning, using default loading method")
            model = super().load_from_checkpoint(checkpoint_path, **kwargs)
        return model

    def configure_optimizers(self):
        params = [{'params': self.latent_codes.parameters(), 'lr': self.config['train']['lr'], 'name': 'latent_codes'}]
        optimizer = optim.AdamW(params, betas=self.config['train']['betas'])
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, **self.config['train']['lr_scheduler'])
        return [optimizer], [lr_scheduler]

    def dataloader(self, split, shuffle=None):
        return super().dataloader(split, shuffle=False)

    def volume_rendering(self,
                         ray_origins,
                         ray_directions,
                         query_points,
                         depth_values,
                         code,
                         depth_samples_per_ray,
                         extra_model_outputs=[],
                         extra_model_inputs={},
                         fine_sampling=True,
                         randomize=True,
                         white_background=True):

        model_outputs = self.decoder(ray_directions, code, ['sampler'] + extra_model_outputs, extra_model_inputs)
        radiance_field_sampler = model_outputs['sampler']
        del model_outputs['sampler']

        request_sampler_outputs = ['sigma', 'rgb', 'sdf_distance']
        results = defaultdict(list)
        sampler_outputs_coarse = radiance_field_sampler(query_points, request_sampler_outputs)
        for k, v in sampler_outputs_coarse.items():
            results[k].append(v)
        results = {k: torch.cat(v, dim=-3) for k, v in results.items()}

        if fine_sampling:
            sigma = results['sigma']
            rgb = results['rgb']
            query_points_fine, z_indices_sorted, z_values_sorted = fine_sample_rays(
                ray_origins, ray_directions, depth_values, sigma, depth_samples_per_ray, randomize)
            results_fine = defaultdict(list)
            sampler_outputs_fine = radiance_field_sampler(query_points_fine, request_sampler_outputs)
            for k, v in sampler_outputs_fine.items():
                results_fine[k].append(v)
            results_fine = {k: torch.cat(v, dim=-3) for k, v in results_fine.items()}

            sigma_fine = results_fine['sigma']
            rgb_fine = results_fine['rgb']

            sigma = torch.cat((sigma, sigma_fine), dim=-2).gather(
                -2, z_indices_sorted.unsqueeze(-1).expand(-1, -1, -1, sigma.shape[-1]))
            rgb = torch.cat((rgb, rgb_fine), dim=-2).gather(
                -2, z_indices_sorted.unsqueeze(-1).expand(-1, -1, -1, rgb.shape[-1]))
            depth_values = z_values_sorted

        rgb_predicted, depth_predicted, mask_predicted, normals_predicted, semantics_predicted = nerf_utils.render_volume_density(
            sigma.squeeze(-1),
            rgb,
            ray_origins,
            ray_directions,
            depth_values,
            white_background=white_background)

        results = {
            'rgb': rgb_predicted.squeeze(0) / 2 + 0.5,  # [n_rays, 3]
            'depth': depth_predicted.squeeze(0),
            'mask': mask_predicted.squeeze(0).unsqueeze(1),
            'min_sdf': results['sdf_distance'].squeeze(0).squeeze(-1).min(-1)[0],
            'model_outputs': model_outputs
        }
        return results

    def novel_view_synthesis(self,
                             camera,
                             code,
                             ray_multiplier=1,
                             res_multiplier=1,
                             extra_model_outputs=[],
                             extra_model_inputs={},
                             force_no_cam_grad=False,
                             backward_fcn=None):
        height, width = int(camera['height'] * res_multiplier), int(camera['width'] * res_multiplier),
        depth_samples_per_ray = self.config['depth_samples_per_ray'] * ray_multiplier
        tform_cam2world = camera['cam2world_mat'].unsqueeze(0)
        focal = camera['K'][0, 0] / 128
        ray_origins, ray_directions, near_thresh, far_thresh = sample_rays(height, width, focal, tform_cam2world, scene_range=self.decoder.scene_range)
        query_points, depth_values = sample_points_from_rays(ray_origins,
                                                             ray_directions,
                                                             near_thresh,
                                                             far_thresh,
                                                             depth_samples_per_ray,
                                                             randomize=True)
        if force_no_cam_grad:
            query_points = query_points.detach()
            depth_values = depth_values.detach()
            ray_directions = ray_directions.detach()

        output = self.volume_rendering(
            ray_origins,
            ray_directions,
            query_points,
            depth_values,
            code,
            depth_samples_per_ray,
            extra_model_outputs=extra_model_outputs,
            extra_model_inputs=extra_model_inputs)
        output['rgb'] = output['rgb'].reshape(height, width, 3)
        output['mask'] = output['mask'].reshape(height, width, 1)
        if backward_fcn is not None:
            loss = self.loss(output, {'rgb': camera['image']['color']})
            if loss:
                backward_fcn(loss)
                output['loss'] = self.loss.compute_metrics()
        return output

    def optimize_step(self, opt, objnerf, manual_backward_fcn=None):
        if manual_backward_fcn is None:
            manual_backward_fcn = self.manual_backward
            opt.zero_grad()

        results = {}
        for camera_id, camera in objnerf['camera'].items():
            results[camera_id] = self.novel_view_synthesis(
                camera,
                objnerf['latent_code'] if 'latent_code' in objnerf else self.latent_codes(objnerf['idx'])[None, None, ...].repeat(
                    1, self.decoder.mapping_network.backbone.num_ws, 1),  # update latent code
                force_no_cam_grad=True,
                extra_model_outputs=['attention_values'],
                backward_fcn=manual_backward_fcn
            )

        if manual_backward_fcn is not None:
            opt.step()

        return results

    def mean_latent_code(self, category_id):
        if not isinstance(category_id, torch.Tensor):
            category_id = torch.tensor(category_id, device=self.device)
        label = category_id.reshape(-1).clone().detach()
        label = self.decoder.class_embedding(label) if self.decoder.num_classes else None
        return self.decoder.mapping_network.get_average_w(label)

    def training_step(self, batch, batch_idx):
        # optimize the latent codes
        objnerf = ObjectNeRF(batch, backend=torch, device=self.device)
        metrics = self.optimize_latent_code(
            objnerf, self.config['train'], self.optimizers(), self.lr_schedulers(), get_video=objnerf['if_log'])
        if objnerf['if_log']:
            metrics, opt_video = metrics
            self.logger.experiment.log({'global_step': batch_idx, 'train/optimization/video': opt_video})

        # reset optimizer and lr scheduler
        self.trainer.strategy.setup_optimizers(self.trainer)

    def optimize_latent_code(self, objnerf, config, optimizer, lr_scheduler, get_video=False):
        # start optimization
        logs = defaultdict(list)
        generated_imgs = []
        for i in range(config['num_opts']):
            results = self.optimize_step(optimizer, objnerf)
            lr_scheduler.step()
            if get_video:
                generated_img = image_float_to_uint8(next(iter(results.values()))['rgb'].detach().cpu().numpy())
                gt_img = image_float_to_uint8(next(iter(objnerf['camera'].values()))['image']['color'].cpu().numpy())
                combined_img = image_grid([generated_img, gt_img], short_height=True)
                generated_imgs.append(combined_img.transpose(2, 0, 1))
            for k, v in results[0]['loss'].items():
                logs[k].append(v)

        # log optimization process as video
        if get_video:
            fps = min(len(generated_imgs) / config['min_video_length'], 24)
            generated_imgs = np.stack(generated_imgs)
            video = wandb.Video(generated_imgs, fps=fps, format="gif", caption=objnerf['jid'])
            return logs, video
        return logs

    def eval_given_latent_code(self, batch, config, get_video=False):
        generated_imgs = []
        metrics = defaultdict(list)
        for camera_id, camera in batch['camera'].items():
            generated_img = self.novel_view_synthesis(camera, batch['latent_code'],
                                                      force_no_cam_grad=True,
                                                      extra_model_outputs=['attention_values'])['rgb']
            metrics['psnr'].append(torchmetrics.functional.peak_signal_noise_ratio(
                generated_img, camera['image']['color'], data_range=1.0))
            metrics['ssim'].append(torchmetrics.functional.structural_similarity_index_measure(
                generated_img.permute(2, 0, 1).unsqueeze(0),
                camera['image']['color'].permute(2, 0, 1).unsqueeze(0),
                data_range=1.0
            ))

            if get_video:
                generated_img = image_float_to_uint8(generated_img.cpu().numpy())
                gt_img = image_float_to_uint8(camera['image']['color'].cpu().numpy())
                combined_img = image_grid([generated_img, gt_img], short_height=True)
                generated_imgs.append(combined_img.transpose(2, 0, 1))
        metrics = {k: torch.mean(torch.stack(v)) for k, v in metrics.items()}

        # log output frames as video
        if get_video:
            fps = min(len(generated_imgs) / config['min_video_length'], 24)
            generated_imgs = np.stack(generated_imgs)
            video = wandb.Video(generated_imgs, fps=fps, format="gif", caption=batch['jid'])
            return metrics, video
        return metrics

    def test_step(self, batch, batch_idx):
        # sample camera
        objnerf = ObjectNeRF(batch, backend=torch, device=self.device)

        # init latent code
        latent_code = self.mean_latent_code(objnerf['category_id'])

        # First optimize the latent codes
        with torch.enable_grad(), torch.inference_mode(False):
            objnerf_optim = objnerf.subset(self.config['test']['optimize_camera_id']) if self.config['test']['optimize_camera_id'] else objnerf
            objnerf_optim['latent_code'] = latent_code.clone().requires_grad_()
            optimizer = torch.optim.Adam([objnerf_optim['latent_code']], lr=self.config['test']['lr'], betas=self.config['test']['betas'])
            lr_scheduler = optim.lr_scheduler.StepLR(optimizer, **self.config['test']['lr_scheduler'])
            metrics = self.optimize_latent_code(objnerf_optim, self.config['test'], optimizer, lr_scheduler, get_video=batch['if_log'])
        objnerf['latent_code'] = objnerf_optim['latent_code'].detach().clone()
        if batch['if_log']:
            metrics, opt_video = metrics
            self.logger.experiment.log({'global_step': batch_idx, 'test/optimization/video': opt_video})

            # log optimization video into wandb table
            test_sample_row = {'jid': objnerf['jid'], 'optimization/video': opt_video}
            # log optimization curve into wandb table
            iterations = list(range(self.config['test']['num_opts']))
            for metric, values in metrics.items():
                plt.plot(iterations, values)
                plt.xlabel("iteration")
                plt.ylabel(metric)
                test_sample_row[f"optimization/{metric}"] = wandb.Image(plt)
                plt.close()

        # Then evaluate
        metrics = self.eval_given_latent_code(objnerf, self.config['test'], get_video=batch['if_log'])
        if batch['if_log']:
            metrics, eval_video = metrics
            self.logger.experiment.log({'global_step': batch_idx, 'test/evaluation/video': eval_video})

            test_sample_row['evaluation/video'] = eval_video
            test_sample_row.update(metrics)

            if not hasattr(self, 'test_sample_table'):
                self.test_sample_table = wandb.Table(columns=list(test_sample_row.keys()))
            self.test_sample_table.add_data(*test_sample_row.values())

        # log metrics
        for k, v in metrics.items():
            for category in ['mean', objnerf['category']]:
                self.test_metrics[category][k].update(v)

    def collect_test_metrics(self):
        columns = list(self.test_metrics.keys())
        table = wandb.Table(columns=['metric'] + columns)
        for metric in next(iter(self.test_metrics.values())).keys():
            row = [self.test_metrics[k][metric].compute() for k in columns]
            table.add_data(metric, *row)
        return table

    def on_test_end(self):
        self.logger.experiment.log({'test/evaluation/metrics': self.collect_test_metrics()})
        self.logger.experiment.log({'test/samples': self.test_sample_table})

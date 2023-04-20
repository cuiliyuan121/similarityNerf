import wandb
import torch
from torch import optim, nn
from utils.torch_utils import MyLightningModule, LossModule, get_resnet, initialize_cls
from utils.dataset import CategoryMapping, ObjectNeRF
from external.nerf_from_image.models.encoder import BootstrapEncoder
from .style_nerf import StyleNeRF, StyleNeRFDataset
from torchvision import transforms
import pytorch_lightning as pl
import random
from utils.visualize_utils import image_grid, image_float_to_uint8
from collections import defaultdict
import os
import numpy as np
from PIL import Image
import yaml
from utils.general_utils import recursive_merge_dict


class AutoEncoderNeRFDataset(StyleNeRFDataset):
    def __init__(self, config, split):
        super().__init__(config, split)
        self.data_dict_keys.append('category_id')
        
        # load object crops if specified
        if config['input_dir'] is not None:
            object_crop_ids, _ = self.load_split(config['input_dir'])
            
            # find intersection of objects in object_crop and object data split
            object_scene_ids = defaultdict(list)
            self.object_id2idx = {obj_id: idx for idx, obj_id in enumerate(self.object_ids)}
            for object_crop_id in object_crop_ids:
                object_id, scene_id = object_crop_id.split('.')
                if object_id in self.object_id2idx:
                    object_scene_ids[object_id].append(scene_id)
            self.object_crop_ids = [f"{k}.{v}" for k, l in object_scene_ids.items() for v in l]

            print(f"{len(object_scene_ids)} of them have image crops from scenes.")
            print(f"Got {len(self.object_crop_ids)} object-scene combinations for {split} split")

        # initialize transform
        self.train_input_transform = transforms.Compose([
            transforms.RandomCrop((config['random_crop_width'], config['random_crop_width'])),
            transforms.RandomHorizontalFlip()
        ])

    def __len__(self):
        return len(self.object_crop_ids)

    def __getitem__(self, idx):
        object_crop_id = self.object_crop_ids[idx]
        gt_data = super().__getitem__(self.object_id2idx[object_crop_id.split('.')[0]])

        # load object crop
        object_dir = os.path.join(self.config['input_dir'], object_crop_id)
        object_crop = self.load_and_sample_object(object_dir, idx)

        # random crop and normalize color image
        for camera in object_crop['camera'].values():
            resize_width = self.config['resize_width'] if self.split == 'train' else self.config['random_crop_width']
            nerf_enc_input = camera['image'].preprocess_image(self.config['input_images'], resize_width)
            if self.split == 'train':
                nerf_enc_input = self.train_input_transform(nerf_enc_input)
            camera['image']['nerf_enc_input'] = nerf_enc_input

        # get data as a dict
        input_data = object_crop.data_dict(
            keys=['category_id', 'camera', 'jid'],
            camera={'keys': {'image': {'keys': ['color', 'nerf_enc_input']}}}
        )

        return input_data, gt_data


class LatentEncoder(pl.LightningModule):
    def __init__(self, backbone, n_layers, category_emdedding, W, latent_dim, num_category, pretrained=False, input_images=None):
        super().__init__()
        self.save_hyperparameters()

        # initialize feature embedding
        self.resnet, n_inputs = get_resnet(backbone, pretrained=pretrained, input_images=input_images)

        # initialize category embedding
        self.category_embedding = nn.Embedding(num_category, category_emdedding)

        # initialize MLP
        layers = []
        in_dim, out_dim = n_inputs + category_emdedding, W
        for i in range(n_layers):
            if i == n_layers - 1:
                out_dim = latent_dim
            layers.append(nn.Linear(in_dim, out_dim))
            if i < n_layers - 1:
                layers.append(nn.LeakyReLU(0.2, True))
            in_dim = out_dim
        mlp = nn.Sequential(*layers)
        setattr(self, 'mlp', mlp)

    def get_image_feature(self, image):
        return self.resnet(image)

    def latent_code_from_image_feature(self, image_feature, category_id):
        x = torch.cat([image_feature, self.category_embedding(category_id)], dim=-1)
        return self.mlp(x).unsqueeze(1)

    def forward(self, image, category_id):
        image_feature = self.get_image_feature(image)
        return self.latent_code_from_image_feature(image_feature, category_id)


class AutoEncoderNeRFLoss(LossModule):
    def __init__(self, func=None, weight=None, **kwargs):
        assert 'latent_code' in func and 'latent_code' in weight
        super().__init__(func, weight, **kwargs)

    def compute_loss(self, est, gt):
        loss = 0.

        if self.weight['latent_code'] is not None:
            latent_code_loss = self.func['latent_code'](est, gt)
            self.metrics['latent_code_loss'].update(latent_code_loss)
        if self.weight['latent_code']:
            loss += self.weight['latent_code'] * latent_code_loss

        return loss


class AutoEncoderNeRF(MyLightningModule):
    dataset_cls = AutoEncoderNeRFDataset

    def __init__(self, encoder_module, encoder, decoder, input_images=None, loss=None, **kwargs):
        super().__init__(**kwargs)
        self.automatic_optimization = False

        # set input images
        input_images = input_images or ['color_norm', 'mask']
        self.config['dataset']['input_images'] = input_images

        if config_dir := decoder.pop('config_dir', None):
            print(f"Reading config file from {config_dir} for StyleNeRF")
            with open(config_dir, 'r') as f:
                config = yaml.load(f, Loader=yaml.FullLoader)
            decoder = recursive_merge_dict(config['model'], decoder)
        if checkpoint_dir := decoder.pop('checkpoint', None):
            self.decoder = StyleNeRF.load_from_checkpoint(checkpoint_dir, **decoder)
            decoder.update(self.decoder.hparams)
        else:
            self.decoder = initialize_cls(StyleNeRF, **decoder)

        # initialize encoder
        if encoder is not None:
            if encoder_module == 'LatentEncoder':
                if encoder['num_category'] == 'auto':
                    encoder['num_category'] = len(CategoryMapping.object_categories)
                if 'latent_dim' not in encoder:
                    encoder['latent_dim'] = self.decoder.hparams.decoder['latent_dim']
            ckpt_dir = encoder.pop('checkpoint', None)
            self.encoder = initialize_cls(globals()[encoder_module], **encoder)
            if ckpt_dir:
                assert encoder_module == 'BootstrapEncoder', "Only BootstrapEncoder supports loading from checkpoint"
                checkpoint = torch.load(ckpt_dir)
                state_dict = {k.strip('module.'): v for k, v in checkpoint['model_coord'].items()}
                self.encoder.load_state_dict(state_dict)

        # initialize loss
        if loss and 'func' in loss:
            self.loss = AutoEncoderNeRFLoss(**loss)

        self.save_hyperparameters()

    def configure_optimizers(self):
        params = [{'params': self.encoder.parameters(), 'lr': self.config['train']['lr']['encoder'], 'name': 'latent_encoder'}]
        decoder_lr = self.config['train']['lr']['decoder']
        if decoder_lr is not None:
            params.append({'params': self.decoder.decoder.parameters(), 'lr': decoder_lr, 'name': 'nerf_decoder'})
        optimizer = optim.AdamW(params)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, **self.config['train']['lr_scheduler'])
        return [optimizer], [lr_scheduler]

    def training_step(self, batch, batch_idx):
        input_data, gt_data = batch
        wandb_logger = self.logger.experiment
        opt = self.optimizers()

        est_code = self.embed_latent_code(input_data)

        if self.config['train']['lr']['decoder'] is None:
            opt.zero_grad()
            gt_codes = self.decoder.latent_codes(gt_data['idx'])
            loss = self.loss(est_code.squeeze(), gt_codes)
            self.manual_backward(loss)
            opt.step()
            log_dict = self.loss.compute_metrics()
        else:
            gt_data.update(est_code)
            results = self.decoder.optimize_step(opt, gt_data, manual_backward_fcn=self.manual_backward)
            log_dict = {loss: np.mean([results[k]['loss'][loss] for k in results]) for loss in results[0]['loss']}
            if (self.config['train']['log_img_iter'] > 0) and (self.global_step % self.config['train']['log_img_iter'] == 0):
                input_color = image_float_to_uint8(input_data['camera'][0]['image']['color'].cpu().numpy())
                generated_img = image_float_to_uint8(results[0]['rgb'].cpu().numpy())
                gt_img = image_float_to_uint8(gt_data['camera'][0]['image']['color'].cpu().numpy())
                input_color = np.array(Image.fromarray(input_color).resize(generated_img.shape[:2][::-1]))
                combined_img = image_grid([input_color, generated_img, gt_img], rows=1)
                log_dict["generated_img"] = wandb.Image(combined_img, caption=input_data['jid'])

        log_dict = {f"train/{k}": v for k, v in log_dict.items()}
        log_dict['global_step'] = self.global_step
        wandb_logger.log(log_dict)

    def on_train_epoch_end(self):
        self.lr_schedulers().step()

    def embed_image_feature(self, objnerf):
        input_images = torch.stack([c['image']['nerf_enc_input'] for c in objnerf['camera'].values()])
        image_feature = self.encoder.get_image_feature(input_images)
        return torch.mean(image_feature, dim=0, keepdim=True)

    def embed_latent_code(self, objnerf):
        image_feature = objnerf.get('image_feature', self.embed_image_feature(objnerf))
        if self.hparams.encoder['num_category']:
            category_id = torch.tensor(objnerf['category_id'], device=self.device).unsqueeze(0)
        else:
            category_id = None
        return self.encoder.latent_code_from_image_feature(image_feature, category_id)

    def validation_step(self, batch, batch_idx):
        input_data, gt_data = batch
        wandb_logger = self.logger.experiment

        gt_data['latent_code'] = self.embed_latent_code(input_data)
        metrics = self.decoder.eval_given_latent_code(gt_data, self.config['test'], get_video=input_data['if_log'])
        if input_data['if_log']:
            metrics, video = metrics
            input_color = next(iter(input_data['camera'].values()))['image']['color'].cpu().numpy()
            wandb_logger.log({
                'val/input': wandb.Image(image_float_to_uint8(input_color), caption=input_data['jid']),
                'val/video': video, 'global_step': self.global_step, 'epoch': self.current_epoch
            })
        for k, v in metrics.items():
            self.decoder.val_metrics[k].update(v)

    def on_validation_epoch_end(self):
        if self.global_step > 0:
            log_dict = {f"val/{k}": v.compute() for k, v in self.decoder.val_metrics.items() if v._update_called}
            log_dict['global_step'], log_dict['epoch'] = self.global_step, self.current_epoch
            self.logger.experiment.log(log_dict)
        for m in self.decoder.val_metrics.values():
            m.reset()

    def on_test_start(self):
        self.decoder.test_sample_table = wandb.Table(columns=[
            "jid", "evaluation/video", "evaluation/psnr", "evaluation/ssim"])
        self.camera_id_rng = random.Random(0)

    def test_step(self, batch, batch_idx):
        input_data, gt_data = batch
        test_sample_row = [input_data['jid']]

        # First embed latent codes from input image
        gt_data['latent_code'] = self.embed_latent_code(input_data)

        # Then evaluate
        metrics, video = self.decoder.eval_given_latent_code(gt_data, self.config['test'], get_video=True)
        test_sample_row.append(video)
        test_sample_row.extend(metrics.values())
        for k, v in metrics.items():
            for category in ['mean', gt_data['category']]:
                self.decoder.test_metrics[category][k].update(v)

        self.decoder.test_sample_table.add_data(*test_sample_row)

        # log to wandb
        self.logger.experiment.log({'global_step': batch_idx, 'test/evaluation/video': video})

    def on_test_end(self):
        self.logger.experiment.log({'test/evaluation/metrics': self.decoder.collect_test_metrics()})
        self.logger.experiment.log({'test/samples': self.decoder.test_sample_table})

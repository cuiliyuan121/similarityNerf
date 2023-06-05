import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_convert
import pytorch_lightning as pl
import os
import tempfile
from utils.general_utils import recursive_merge_dict
from utils.dataset import CategoryMapping
import torchmetrics
from external.objsdf_general_utils import get_class
import numpy as np
from pytorch_lightning.callbacks import LearningRateMonitor


def recursive_apply_on_tensor(data, func):
    if isinstance(data, torch.Tensor):
        return func(data)
    if isinstance(data, dict):
        return {key: recursive_apply_on_tensor(value, func) for key, value in data.items()}
    if isinstance(data, list):
        return [recursive_apply_on_tensor(value, func) for value in data]
    if isinstance(data, tuple):
        return tuple(recursive_apply_on_tensor(value, func) for value in data)
    return data


def single_batch_collate(batch):
    assert len(batch) == 1, f"Batch size must be 1, but got {len(batch)}"
    return default_convert(batch[0])


@torch.jit.script
def tensor_linspace(start, end, steps: int):
    """
    Vectorized version of torch.linspace.
    https://github.com/zhaobozb/layout2im/blob/master/models/bilinear.py#L246
    Inputs:
    - start: Tensor of any shape
    - end: Tensor of the same shape as start
    - steps: Integer
    Returns:
    - out: Tensor of shape start.size() + (steps,), such that
    out.select(-1, 0) == start, out.select(-1, -1) == end,
    and the other elements of out linearly interpolate between
    start and end.
    """
    assert start.size() == end.size()
    view_size = start.size() + (1,)
    w_size = (1,) * start.dim() + (steps,)
    out_size = start.size() + (steps,)

    start_w = torch.linspace(1, 0, steps=steps, device=start.device)
    start_w = start_w.view(w_size).expand(out_size)
    end_w = torch.linspace(0, 1, steps=steps, device=start.device)
    end_w = end_w.view(w_size).expand(out_size)

    start = start.contiguous().view(view_size).expand(out_size)
    end = end.contiguous().view(view_size).expand(out_size)

    out = start_w * start + end_w * end
    return out


class LossModule(nn.Module):
    def __init__(self, func, weight, log_loss_key='total_loss', **kwargs):
        super().__init__()

        # initialize metric for each loss
        self.weight = weight
        self.func = nn.ModuleDict({k: get_class(v)() for k, v in func.items()} if func else {})
        self.metrics = nn.ModuleDict({f"{k}_loss": torchmetrics.MeanMetric() for k in weight.keys() if k is not None})

        # log total loss with log_loss_key
        self.log_loss_key = log_loss_key
        if log_loss_key:
            self.metrics[log_loss_key] = torchmetrics.MeanMetric()

        # save initial weights for run time modification
        self.init_weight = weight.copy()

        if kwargs:
            print(f"Warning: {kwargs} is not used in {self.__class__.__name__}")

    def compute_metrics(self):
        metrics = {k: v.compute().item() for k, v in self.metrics.items() if v._update_called}
        for m in self.metrics.values():
            m.reset()
        metrics = {k: v for k, v in metrics.items() if not np.isnan(v)}
        return metrics

    def forward(self, *args, log=True, **kwargs):
        if not log:
            old_update_fcn = {}
            for k, m in self.metrics.items():
                old_update_fcn[k] = m.update
                m.update = lambda *args, **kwargs: None

        loss = self.compute_loss(*args, **kwargs)
        if self.log_loss_key:
            self.metrics[self.log_loss_key].update(loss)

        if not log:
            for k, m in self.metrics.items():
                m.update = old_update_fcn[k]
        return loss

    def compute_loss(self):
        raise NotImplementedError

    def reset_weight(self):
        self.weight = self.init_weight.copy()

    def update_weight_based_on_init(self, weight=True):
        if weight is True:
            self.reset_weight()
        elif weight is False:
            self.weight = {k: None for k in self.weight.keys()}
        else:
            self.weight.update(weight)

    def is_enabled(self):
        return any(self.weight.values())

    def disable_loss(self, key):
        self.weight[key] = None
        self.init_weight[key] = None


def dpp_global_step(global_step):
    return (global_step - 1) * int(os.environ.get('WORLD_SIZE', 1)) + int(os.environ.get('LOCAL_RANK', 0))


class MyLightningModule(pl.LightningModule):
    dataset_cls = None

    def __init__(self, **kwargs):
        super().__init__()
        self.config = kwargs
        if self.config.get('dataset', {}).get('dir'):
            CategoryMapping.load_category_mapping(self.config['dataset']['dir'])

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, strict=True, **kwargs):
        print("Overwriting hparams of checkpoint")
        checkpoint = torch.load(checkpoint_path)
        checkpoint['hyper_parameters'] = recursive_merge_dict(checkpoint['hyper_parameters'], kwargs)
        with tempfile.NamedTemporaryFile() as f:
            torch.save(checkpoint, f)
            return super().load_from_checkpoint(f.name, strict=strict)

    @property
    def global_step(self):
        global_step = self.trainer.global_step if self._trainer else 1
        return dpp_global_step(global_step)

    def dataset(self, split):
        print(f"Constructing {split} dataset")
        if hasattr(self, f"_{split}_dataset"):
            return getattr(self, f"_{split}_dataset")
        dataset = self.dataset_cls(self.config['dataset'], split)
        setattr(self, f"_{split}_dataset", dataset)
        return dataset

    def dataloader(self, split, shuffle=None):
        if shuffle is None:
            shuffle = split == 'train'
        data_loader = DataLoader(
            self.dataset(split), shuffle=shuffle, **self.config['dataloader'],
            collate_fn=single_batch_collate)
        return data_loader

    def train_dataloader(self):
        return self.dataloader('train')

    def val_dataloader(self):
        return self.dataloader('val')

    def test_dataloader(self):
        return self.dataloader('test')

    def predict_dataloader(self):
        return self.dataloader('predict')

    def on_train_batch_start(self, batch, batch_idx):
        self._update_batch_log_info('train', batch, self.global_step)

    def on_validation_batch_start(self, batch, batch_idx, dataloader_idx):
        self._update_batch_log_info('val', batch, batch_idx)

    def on_test_batch_start(self, batch, batch_idx, dataloader_idx):
        self._update_batch_log_info('test', batch, batch_idx)

    def on_predict_batch_start(self, batch, batch_idx, dataloader_idx):
        self._update_batch_log_info('predict', batch, batch_idx)

    def _update_batch_log_info(self, stage, batch, step):
        if_log = self.config.get(stage, {}).get('log_img_iter', False) and step % self.config[stage]['log_img_iter'] == 0
        if isinstance(batch, dict):
            batch = [batch]
        for b in batch:
            b['if_log'] = if_log
            b['batch_idx'] = step


class MyLearningRateMonitor(LearningRateMonitor):
    def _extract_stats(self, trainer, interval):
        latest_stat = super()._extract_stats(trainer, interval)
        latest_stat['global_step'] = dpp_global_step(trainer.global_step)
        latest_stat['epoch'] = trainer.current_epoch
        return latest_stat


def get_resnet(backbone, pretrained, input_images=None):
    # initialize ResNet backbone
    resnet = torch.hub.load('pytorch/vision:v0.10.0', backbone, pretrained=pretrained)
    dim_output = resnet.fc.in_features
    resnet.fc = nn.Identity()

    # change input channels if input_images
    if input_images is not None:
        input_image_channels = {'color_norm': 3, 'mask': 1, 'depth': 1}
        input_channels = sum(input_image_channels[img] for img in input_images)
        if input_channels != 3:
            resnet.conv1 = nn.Conv2d(
                input_channels,
                resnet.conv1.out_channels,
                kernel_size=resnet.conv1.kernel_size,
                stride=resnet.conv1.stride,
                padding=resnet.conv1.padding,
                bias=resnet.conv1.bias is not None
            )

    return resnet, dim_output


def initialize_cls(cls, *args, **kwargs):
    '''Helper function to avoid bugs with pytorch-lightning save_hyperparameters'''
    return cls(*args, **kwargs)

import torch
import torch.nn.functional as F
import numpy as np


def augment_impl(img, pose, focal, p, disable_scale=False, cached_tform=None):
    bs = img.shape[0] if img is not None else pose.shape[0]
    device = img.device if img is not None else pose.device

    if cached_tform is None:
        rot = (torch.rand((bs,), device=device) - 0.5) * 2 * np.pi
        rot = rot * (torch.rand((bs,), device=device) < p).float()

        if disable_scale:
            scale = torch.ones((bs,), device=device)
        else:
            scale = torch.exp2(torch.randn((bs,), device=device) * 0.2)
            scale = torch.lerp(torch.ones_like(scale), scale, (torch.rand(
                (bs,), device=device) < p).float())

        translation = torch.randn((bs, 2), device=device) * 0.1
        translation = torch.lerp(torch.zeros_like(translation), translation,
                                 (torch.rand(
                                     (bs, 1), device=device) < p).float())

        cached_tform = rot, scale, translation
    else:
        rot, scale, translation = cached_tform

    mat = torch.zeros((bs, 2, 3), device=device)
    mat[:, 0, 0] = torch.cos(rot)
    mat[:, 0, 1] = -torch.sin(rot)
    mat[:, 0, 2] = translation[:, 0]
    mat[:, 1, 0] = torch.sin(rot)
    mat[:, 1, 1] = torch.cos(rot)
    mat[:, 1, 2] = -translation[:, 1]
    if img is not None:
        mat_scaled = mat.clone()
        mat_scaled *= scale[:, None, None]
        mat_scaled[:, :, 2] = torch.sum(mat[:, :2, :2] *
                                        mat_scaled[:, :, 2].unsqueeze(-2),
                                        dim=-1)
        grid = F.affine_grid(mat_scaled, img.shape, align_corners=False)
        img = img - 1  # Adjustment for white background
        img_transformed = F.grid_sample(img,
                                        grid,
                                        mode='bilinear',
                                        padding_mode='zeros',
                                        align_corners=False)
        img_transformed = img_transformed + 1  # Adjustment for white background
    else:
        img_transformed = None

    return img_transformed, pose, focal, cached_tform


def augment(img,
            pose,
            focal,
            p,
            disable_scale=False,
            cached_tform=None,
            return_tform=False):
    if p == 0 and cached_tform is None:
        return img, pose, focal

    assert img is None or pose is None or img.shape[0] == pose.shape[0]

    # Standard augmentation
    img_new, pose_new, focal_new, tform = augment_impl(img, pose, focal, p,
                                                       disable_scale,
                                                       cached_tform)

    if return_tform:
        return img_new, pose_new, focal_new, tform
    else:
        return img_new, pose_new, focal_new

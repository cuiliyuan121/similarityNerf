import numpy as np


compatible_backends = {}
ignore_device_funcs = ['array', 'zeros', 'ones', 'eye']


def get_backend(array):
    if isinstance(array, np.ndarray):
        if 'numpy' not in compatible_backends:
            def ignore_device(func):
                return lambda *args, device=None, **kwargs: func(*args, **kwargs)
            for func in ignore_device_funcs:
                setattr(np, f"{func}_on", ignore_device(getattr(np, func)))
            compatible_backends['numpy'] = np
        return compatible_backends['numpy']

    if 'torch' not in compatible_backends:
        import torch
        func_alias = {f"{func}_on": func for func in ignore_device_funcs}
        func_alias.update({'concatenate': 'cat', 'copy': 'clone', 'array_on': 'tensor'})
        for alias, func in func_alias.items():
            setattr(torch, alias, getattr(torch, func))
        torch.isscalar = lambda x: not isinstance(x, torch.Tensor) or x.numel() == 1
        compatible_backends['torch'] = torch
    return compatible_backends['torch']


def homotrans(M, p):
    if p.shape[-1] == M.shape[1] - 1:
        backend = get_backend(p)
        p = backend.concatenate([p, backend.ones_like(p[..., :1])], -1)
    p = p @ M.T
    p = p[..., :-1] / p[..., -1:]
    return p


def cam2uv(K, p):
    p = get_backend(p).copy(p)
    p[..., 0] *= -1
    uv = homotrans(K, p)
    return uv


def uv2cam(K, uv, depth=None, dis=None):
    assert depth is None or dis is None, "depth and dis can not be both provided"
    backend = get_backend(K)
    xy = uv - K[:2, 2]
    xy[..., 1] *= -1
    normalized_xy = xy / K[(0, 1), (0, 1)]
    if depth is None:
        depth = dis / backend.norm(backend.cat([normalized_xy, backend.ones_like(xy[..., :1])], -1), dim=-1)
    depth = backend.ones_like(xy[..., :1]) * depth if backend.isscalar(depth) else depth[..., None]
    xy = normalized_xy * depth
    p = backend.concatenate([xy, -depth], -1)
    return p


def is_in_cam(K, width, height, p):
    depth = -p[..., -1]
    uv = cam2uv(K, p)
    width_height = get_backend(uv).array_on([width, height], device=getattr(uv, 'device', None))
    in_cam = ((uv <= width_height) & (uv >= 0)).all(-1) & (depth > 0)
    return in_cam


def bdb3d_corners(obj):
    corners = np.unpackbits(np.arange(8, dtype=np.uint8)[..., np.newaxis],
                            axis=1, bitorder='little', count=-5).astype(np.float32)
    corners = corners - 0.5
    if obj.backend.__name__ == 'torch':
        corners = obj.backend.tensor(corners, device=obj['size'].device)
    corners = homotrans(obj['local2world_mat'], corners * obj['size'])
    return corners


def bbox_from_binary_mask(binary_mask):
    backend = get_backend(binary_mask)
    # Find all columns and rows that contain 1s
    rows = binary_mask.any(1)
    cols = binary_mask.any(0)
    # Find the min and max col/row index that contain 1s
    rmin, rmax = backend.where(rows)[0][[0, -1]]
    cmin, cmax = backend.where(cols)[0][[0, -1]]
    # Calc height and width
    h = rmax - rmin + 1
    w = cmax - cmin + 1
    return [int(cmin), int(rmin), int(w), int(h)]


BDB3D_FACES = np.array([
    [0, 4, 6], [0, 6, 2], [3, 2, 6], [3, 6, 7],
    [7, 6, 4], [7, 4, 5], [5, 1, 3], [5, 3, 7],
    [1, 0, 2], [1, 2, 3], [5, 4, 0], [5, 0, 1]
])


def crop_image(image, bdb2d):
    cmin, rmin, w, h = bdb2d
    return image[rmin: rmin + h, cmin: cmin + w, ...]


def crop_image_chw(image, bdb2d):
    cmin, rmin, w, h = bdb2d
    return image[..., rmin: rmin + h, cmin: cmin + w]


def crop_K(K, bdb2d):
    K = get_backend(K).copy(K)
    K[:2, 2] -= bdb2d[:2]
    return K


def crop_camera(camera, obj2d):
    camera = camera.copy()
    del camera['object']
    bdb2d = obj2d['bdb2d']

    # crop image according to 2d bbox
    image_io = camera['image']
    for key in image_io:
        if key == 'instance_segmap':
            continue
        if key == 'color_norm':
            image_io[key] = crop_image_chw(image_io[key], bdb2d)
        else:
            image_io[key] = crop_image(image_io[key], bdb2d)
    image_io['instance_segmap'] = crop_image(obj2d['segmentation'], bdb2d)

    # update camera intrinsics for cropped image
    camera['width'], camera['height'] = bdb2d[2:]
    camera['K'] = crop_K(camera['K'], bdb2d)

    return camera


def rotation_mat_dist(mat1, mat2):
    relative_rotation_mat = mat1 @ mat2.T
    return np.rad2deg(np.arccos(np.clip((relative_rotation_mat.trace() - 1) / 2, -1, 1)))


def total3d_world_rotation(rotation_mat):
    backend = get_backend(rotation_mat)
    forward_vec = -rotation_mat[:, 2]
    forward_vec[2] = 0.
    forward_vec = forward_vec / backend.linalg.norm(forward_vec)
    up_vec = backend.array_on([0., 0., 1.], device=getattr(rotation_mat, 'device', None))
    right_vec = backend.cross(forward_vec, up_vec)
    return backend.stack([right_vec, forward_vec, up_vec])


# transform world frame to total3d: X’=Y, Y’=Z, Z’=X
WORLD_FRAME_TO_TOTAL3D = np.array([
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 0]
], dtype=np.float32)


# transform camera frame to total3d: X’=-Z, Y’=Y, Z’=X
CAMERA_FRAME_TO_TOTAL3D = np.array([
    [0, 0, -1],
    [0, 1, 0],
    [1, 0, 0]
], dtype=np.float32)

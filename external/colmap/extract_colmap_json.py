import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import glob  # 查找符合自己要求的文件，
from colmap_utils import read_cameras_binary, read_images_binary, read_points3d_binary

import numpy as np
import os
import argparse
from collections import defaultdict
import json
from tqdm import tqdm


def get_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str,
                        default='/home/yangzesong/Projects/NeRF/datasets/playground',
                        help='root directory of dataset')
    parser.add_argument('--scale_range', type=bool,
                        default=False,
                        help="whether scale the sample range")
    parser.add_argument('--data_type', type=str,
                        default='nerf',
                        help='save as colmap or nerf')  # 主要用于可视化，如果是用于Open3D可视化位姿，则使用colmap，如果用于NeRF训练，则使用nerf

    return vars(parser.parse_args())


def find_cam_param(cam_idx, camdata):
    # 根据提供的图片对应的cam_idx，返回对应的width、height和params
    for cam in camdata:
        if cam_idx == camdata[cam].id:
            if camdata[cam].model == "SIMPLE_PINHOLE" or camdata[cam].model == "SIMPLE_RADIAL":
                params = np.array([camdata[cam].params[0], camdata[cam].params[0],
                                   camdata[cam].params[1], camdata[cam].params[2]])
                return [camdata[cam].width, camdata[cam].height, params]
            return [camdata[cam].width, camdata[cam].height, camdata[cam].params]


def scale_pose(c2ws, scale_factor):
    # 根据尺度对pose的t进行放缩
    scale_c2ws = {}
    for c2w in c2ws:
        scale_c2ws[c2w] = c2ws[c2w].copy()
        scale_c2ws[c2w][:, 3] /= scale_factor
    return scale_c2ws


def extract_json(root_dir, data_type, scale_range=True):
    # 先从image.bin中提取图片的索引，图片对应的idx，图片
    # image.txt:IMAGE_ID, [QW, QX, QY, QZ] , [TX, TY, TZ], CAMERA_ID, NAME
    imdata = read_images_binary(os.path.join(root_dir, 'sparse/0/images.bin'))
    camdata = read_cameras_binary(os.path.join(root_dir, 'sparse/0/cameras.bin'))
    # camera.txt: CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]
    img_info = defaultdict(dict)
    Extrinsics = {}
    World2Cam = {}

    for im_idx in tqdm(imdata):
        img_info[im_idx] = {}
        img = imdata[im_idx]
        img_info[im_idx]["image_name"] = img.name
        img_info[im_idx]["idx"] = img.id
        img_info[im_idx]["cam_idx"] = img.camera_id
        # 根据图片拍摄的对应相机的idx返回其对应的width、height和params
        width, height, intrinsics = find_cam_param(img.camera_id, camdata)
        img_info[im_idx]["width"] = width
        img_info[im_idx]["height"] = height
        img_info[im_idx]["intrinsics"] = intrinsics.tolist()
        # world_2_cam
        R = img.qvec2rotmat()  # 四元数变换为旋转矩阵
        t = img.tvec.reshape(3, 1)
        # w2c = np.concatenate([R, t], 1)
        w2c = np.zeros([4, 4])
        w2c[:3, :3] = R
        w2c[:3, 3] = t.squeeze()
        w2c[3, 3] = 1
        World2Cam[im_idx] = w2c
        # 没有问题

        # cam_2_world
        # *****************************************很重要**********************************************
        transform_matrix = np.linalg.inv(w2c)[:3]
        # Original poses has rotation in form "right down front", change to "right up back"
        # refer the https://github.com/hjxwhy/multinerf
        if data_type == "nerf":
            transform_matrix[:, 1:3] *= -1
        # *****************************************很重要**********************************************

        Extrinsics[im_idx] = transform_matrix
        # img_info[im_idx]["transform_matrix"] = transform_matrix.tolist()

    # 3. 根据坐标点调整尺度
    pts3d = read_points3d_binary(os.path.join(root_dir, 'sparse/0/points3D.bin'))
    # POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]
    xyz_world = np.array([pts3d[p_id].xyz for p_id in pts3d])
    # 齐次坐标系
    xyz_world_h = np.concatenate([xyz_world, np.ones((len(xyz_world), 1))], -1)
    # 找到相机坐标系下最远的far
    nears, fars = {}, {}
    for im_idx in img_info:
        # 将所有的点转换到相机坐标系下
        w2c = World2Cam[im_idx]

        xyz_cam_i = (xyz_world_h @ w2c.T)[:, :3]
        xyz_cam_i = xyz_cam_i[xyz_cam_i[:, 2] > 0]

        nears[im_idx] = np.percentile(xyz_cam_i[:, 2], 0.1)  # 找到分位数
        fars[im_idx] = np.percentile(xyz_cam_i[:, 2], 99.9)

    scale_factor = 1
    if scale_range:
        max_far = np.fromiter(fars.values(), np.float32).max()
        scale_factor = max_far / 5

    Extrinsics_scale = scale_pose(Extrinsics, scale_factor)

    for k in nears:
        nears[k] /= scale_factor
    for k in fars:
        fars[k] /= scale_factor

    # 在json中添加外参和near和far
    for idx in img_info:
        img_info[idx]["transform_matrix"] = Extrinsics_scale[idx].tolist()
        img_info[idx]["near"] = nears[idx]
        img_info[idx]["far"] = fars[idx]
        img_info[idx]["scale_factor"] = scale_factor

    return img_info


if __name__ == "__main__":
    hparams = get_opts()
    print(hparams)

    img_info = extract_json(hparams['root_dir'], data_type=hparams["data_type"], scale_range=hparams["scale_range"])
    print("Finish convert the colmap file into json!")

    os.makedirs(os.path.join(hparams['root_dir'], "json"), exist_ok=True)
    save_path = os.path.join(hparams['root_dir'], "json/transform_info.json")

    imgs = {}
    for idx in tqdm(img_info):
        imgs[idx] = img_info[idx]
        with open(save_path, "w") as fp:
            json.dump(imgs, fp)
            fp.close()

    print("The information has been saved in the path of {0}".format(save_path))

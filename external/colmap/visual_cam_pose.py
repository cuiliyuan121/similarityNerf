import open3d as o3d
import numpy as np
import cv2
import json
import os
from collections import defaultdict


def count_cam_idx(cam_dict):
    indexs = []
    for cam in cam_dict:
        index = cam_dict[cam]["cam_idx"]
        if index not in indexs:
            indexs.append(index)
    return indexs


def get_camera_frustum(img_size, K, C2W, frustum_length, color, scale_pose=True):
    # pose_scale用于放大位姿
    # [w,h]  [4,4]  [3,4]
    W, H = img_size
    hfov = np.rad2deg(np.arctan(W / 2. / K[0, 0]) * 2.)  # 光心到图像左右两边的角度
    vfov = np.rad2deg(np.arctan(H / 2. / K[1, 1]) * 2.)  # 光心到图像上下两边的角度
    half_w = frustum_length * np.tan(np.deg2rad(hfov / 2.))  # 归一化平面
    half_h = frustum_length * np.tan(np.deg2rad(vfov / 2.))
    # 不就是W,H的一半

    pose = np.eye(4)
    pose[:3] = C2W
    C2W = pose

    # 调整尺度
    '''
    half_w *= 5e-3
    half_h *= 5e-3
    frustum_length *= 5e-3
    '''

    # build view frustum for camera (I, 0)
    frustum_points = np.array([[0., 0., 0.],  # frustum origin
                               [-half_w, -half_h, frustum_length],  # 左上角，注意x朝右，y朝上
                               [half_w, -half_h, frustum_length],  # 右上角
                               [half_w, half_h, frustum_length],  # bottom-right image corner
                               [-half_w, half_h, frustum_length],
                               # 坐标轴
                               [0.2, 0, 0], [0, 0.2, 0], [0, 0, 0.2]  # x轴,y轴,z轴
                               ])  # bottom-left image corner
    if scale_pose:
        frustum_points *= 5e-2

    frustum_lines = np.array([[0, i] for i in range(1, 5)] +
                             [[i, (i + 1)] for i in range(1, 4)] +
                             [[4, 1], [0, 5], [0, 6], [0, 7]])
    # 平铺
    frustum_colors = np.tile(np.array(color).reshape((1, 3)), (frustum_lines.shape[0], 1))
    frustum_colors[-1] = np.array([0, 0, 1])  # Z 蓝色
    frustum_colors[-2] = np.array([0, 1, 0])  # y 绿色
    frustum_colors[-3] = np.array([1, 0, 0])  # x 红色

    # frustum_colors = np.vstack((np.tile(np.array([[1., 0., 0.]]), (4, 1)),
    #                            np.tile(np.array([[0., 1., 0.]]), (4, 1))))

    # transform view frustum from (I, 0) to (R, t)
    frustum_points = np.dot(
        np.hstack((frustum_points, np.ones_like(frustum_points[:, 0:1]))),  # 齐次坐标
        C2W.T)  # 8，4
    # 归一化矩阵乘以C2W.T
    frustum_points = frustum_points[:, :3] / frustum_points[:, 3:4]

    return frustum_points, frustum_lines, frustum_colors


def frustums2lineset(frustums):
    N = len(frustums)
    merged_points = np.zeros((N * 8, 3))  # 5 vertices per frustum 总共有5个点
    merged_lines = np.zeros((N * 11, 2))  # 8 lines per frustum # 总共有8条线
    merged_colors = np.zeros((N * 11, 3))  # each line gets a color # 每条线一个颜色

    for i, (frustum_points, frustum_lines, frustum_colors) in enumerate(frustums):
        merged_points[i * 8:(i + 1) * 8, :] = frustum_points
        merged_lines[i * 11:(i + 1) * 11, :] = frustum_lines + i * 8
        merged_colors[i * 11:(i + 1) * 11, :] = frustum_colors

    lineset = o3d.geometry.LineSet()
    lineset.points = o3d.utility.Vector3dVector(merged_points)
    lineset.lines = o3d.utility.Vector2iVector(merged_lines)
    lineset.colors = o3d.utility.Vector3dVector(merged_colors)

    return lineset


def visual_cameras(cam_dicts, data_type="colmap", scale_pose=True):
    # 设置起点在路径中心
    center_frame = cam_dicts[list(cam_dicts.keys())[0]]  # [len(cam_dicts)//2]]
    center_origin = center_frame["c2w"][:, 3]
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.01, origin=list(center_origin))
    things_to_draw = [coord_frame]
    frustums = []

    cam_idxs = count_cam_idx(cam_dicts)
    colors = {}  # 每个index显示一种颜色
    '''
    color_begin = [1, 0, 0]
    color_end = [0, 1, 1]
    color_delta = (np.array(color_end) - np.array(color_begin)) / len(cam_idxs)
    for i,cam in enumerate(cam_idxs):
        colors[cam] = list(
            np.array(color_begin)+i*color_delta
        )
    '''
    for i, cam in enumerate(cam_idxs):
        colors[cam] = list(
            np.random.random(3)
        )

    for cam_dict in cam_dicts:
        cam_info = cam_dicts[cam_dict]
        cam_show_idxs = cam_idxs
        if cam_info["cam_idx"] in cam_show_idxs:  # 筛选index进行展示
            index = cam_info["cam_idx"]

            K = np.array([
                [cam_info["intrinsics"][2], 0, cam_info["intrinsics"][1] / 2, 0],  # 宽
                [0, cam_info["intrinsics"][2], cam_info["intrinsics"][0] / 2, 0],  # 高
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])
            C2W = cam_info["c2w"]

            # NeRF坐标系和open3D不一样，需要转换
            if data_type is not "colmap":
                C2W[:, 1:3] *= -1

            # C2W[:, 1:3] *= -1

            img_size = [cam_info["intrinsics"][1], cam_info["intrinsics"][0]]  # 宽、高
            frustum_length = cam_info["intrinsics"][2] / cam_info["intrinsics"][1]

            frustums.append(get_camera_frustum(img_size, K, C2W, frustum_length, colors[index], scale_pose=scale_pose))

    cameras = frustums2lineset(frustums)
    # o3d.visualization.draw_geometries([cameras])
    things_to_draw.append(cameras)

    o3d.visualization.draw_geometries(things_to_draw)


def average_poses(vis_meta):
    # 提取所有的pose
    poses = []
    for frame in vis_meta:
        poses.append(vis_meta[frame]["c2w"])
    poses = np.stack(poses, 0)
    center = poses[..., 3].mean(0)

    def normalize(v):
        """Normalize a vector."""
        return v / np.linalg.norm(v)

    z = normalize(poses[..., 2].mean(0))  # (3)
    # 3. Compute axis y' (no need to normalize as it's not the final output)
    y_ = poses[..., 1].mean(0)  # (3) # 主要是为了通过叉乘求x
    # 4. Compute the x axis
    x = normalize(np.cross(y_, z))  # (3)
    # 5. Compute the y axis (as z and x are normalized, y is already of norm 1)
    y = np.cross(z, x)  # (3)

    pose_avg = np.stack([x, y, z, center], 1)  # 中心位置相机的位姿

    pose_avg_homo = np.eye(4)
    pose_avg_homo[:3] = pose_avg  # convert to homogeneous coordinate for faster computation 齐次坐标
    # 将center_pose转换成齐次坐标 pose_avg:T_CamCenter2World
    pose_avg_inv = np.linalg.inv(pose_avg_homo)  # 将世界坐标转换为cam_center
    last_row = np.tile(np.array([0, 0, 0, 1]), (len(poses), 1, 1))  # (N_images, 1, 4)
    # pose的齐次坐标
    poses_homo = np.concatenate([poses, last_row], 1)  # (N_images, 4, 4) homogeneous coordinate

    # poses_homo:Camera2World -> pose_avg_inv:World2Center -> Camera2Center
    # poses_homo:Camera2World -> pose_avg_inv:World2Center -> Camera2Center
    poses_centered = pose_avg_inv @ poses_homo  # (N_images, 4, 4)
    poses_centered = poses_centered[:, :3]  # (N_images, 3, 4)

    return poses_centered, pose_avg


if __name__ == "__main__":
    # 首先提取相机pose
    meta = []
    json_path = "/home/yangzesong/Projects/NeRF/datasets/Truck/json/transform_info.json"  # "/home/yangzesong/Projects/NeRF/ngp_pl/data/yangang_Recon/flowerbed/json/transform_info.json"
    # /home/yangzesong/Projects/NeRF/Block-NeRF/Block-NeRF/data/WaymoDataset/json
    data_type = "nerf"  # nerf

    whehter_shrink_pose = False  # 缩减pose

    visual_scale_pose = True  # 可视化缩小pose

    with open(json_path) as fp:
        meta = json.load(fp)

    print(f"Total {len(meta)} frames...")

    if whehter_shrink_pose:
        frames = list(meta.keys())
        new_meta = {}
        indexs = np.arange(len(frames))
        random_index = np.random.choice(indexs, len(meta) // 10)
        print()
        for index in random_index:
            new_meta[frames[index]] = meta[frames[index]]
        meta = new_meta

    c2w_mats = []
    hwfs = []

    vis_meta = defaultdict(dict)

    for frame in meta:
        vis_meta[frame]["c2w"] = np.array(meta[frame]["transform_matrix"])
        vis_meta[frame]["intrinsics"] = [meta[frame]["height"],
                                         meta[frame]["width"],
                                         meta[frame]["intrinsics"][0]]
        vis_meta[frame]["cam_idx"] = meta[frame]["cam_idx"]

    poses_centered, pose_avg = average_poses(vis_meta)

    '''
    for i, frame in enumerate(meta):
        vis_meta[frame]["c2w"] = poses_centered[i]
    '''

    vis_meta["center"] = vis_meta[frame]
    vis_meta["center"]["c2w"] = pose_avg
    visual_cameras(vis_meta, data_type=data_type, scale_pose=visual_scale_pose)

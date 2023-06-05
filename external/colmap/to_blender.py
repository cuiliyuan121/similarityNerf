import sys
sys.path.append('.')
import torch
import yaml
from external.NeRFSceneUnderstanding.models.base import Front3DDataset
from tqdm import tqdm
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import json
from internal.datasets import recenter_poses

class SceneToBlenderDataset(Front3DDataset):
    def __init__(self, stage, dataset_config_dir):
        with open(dataset_config_dir, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        front3d_dataset_config = config['model']['dataset']
        super().__init__(front3d_dataset_config, stage)

    def get_data(self, input_scene, gt_scene):
        # train stage uses gt pose， val/test stage uses estimate pose
        train_camera_idx = input_scene['chosen_cam_id'][self.config['train_image_num']]

        z_nears = []
        z_fars = []

        gt_scene['train_camera_idx'] = train_camera_idx
        for cam_id, camera in gt_scene['camera'].items():
            #camera['image']['color'] = np.array(Image.open(image_path))
            # generate gt mask from object segmentation
            gt_mask = np.zeros((camera['height'], camera['width']), dtype=np.float32)
            for obj_idx, object2d in enumerate(camera['object'].values()):
                gt_mask[object2d['segmentation']] = 1
            camera['image']['mask'] = gt_mask[..., np.newaxis]
            camera['image']['rgba'] = np.concatenate([camera['image']['color'], camera['image']['mask']*255], axis=-1).astype(np.uint8)
            depth_map = camera['image']['depth']
            depth_map = depth_map[camera['image']['mask'][..., 0].astype(np.bool)]
            z_near, z_far = depth_map.min(), depth_map.max()
            z_near = min(0, z_near-0.05)
            z_nears.append(z_near)
            z_fars.append(z_far+0.05)

        gt_scene['z_nears'] = np.array(z_nears).min().astype(np.float32)
        gt_scene['z_fars'] = np.array(z_fars).max().astype(np.float32)

        # get data as a dict
        gt_data = gt_scene.data_dict(
            keys=['uid', 'origin_camera_id','train_camera_idx', 'z_nears', 'z_fars'],
            camera={'image': {'keys': ['color', 'mask', 'rgba']}},
            object={'keys': {'rotation_mat', 'center', 'category_id', 'size', 'jid'}}
        )

        return gt_data


if __name__ == "__main__":
    root_dir = 'external/NeRFSceneUnderstanding'
    dataset_config_dir = os.path.join(root_dir, 'configs/SceneOptimization.yaml')
    train_dataset = SceneToBlenderDataset('test', dataset_config_dir)
    test_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    front3d_blender_dir = 'data/scene-understanding-blender'
    for idx, data in enumerate(tqdm(test_data_loader)):
        train_json = {}
        train_json['near'] = float(data['z_nears'])
        train_json['far'] = float(data['z_fars'])

        train_frames = []
        scene_id = data['uid'][0]
        train_camera_idx = data['train_camera_idx']

        all_poses = np.stack([i['cam2world_mat'].squeeze(0).numpy() for i in data['camera'].values()])
        all_poses = recenter_poses(all_poses)
        for cam_num, (camera_id, camera) in enumerate(data['camera'].items()):
            if camera_id not in train_camera_idx:
                continue
            if 'camera_angle_x' not in train_json:
                width = camera['width']
                focal = camera['K'][0][0][0]
                # self.focal = .5 * self.width / np.tan(.5 * float(meta['camera_angle_x']))
                train_json['camera_angle_x'] = float(2 * np.arctan(width/(2*focal)))

            frames_info = {}
            frames_info['file_path'] = os.path.join('train', f"r_{camera_id:04}")
            os.makedirs(os.path.join(front3d_blender_dir, scene_id, 'train'), exist_ok=True)
            Image.fromarray(camera['image']['rgba'].squeeze(0).numpy(), mode='RGBA').save(os.path.join(front3d_blender_dir,
                                                                           scene_id,'train',f"r_{camera_id:04}.png"))
            frames_info['transform_matrix'] = all_poses[cam_num].tolist()
            train_frames.append(frames_info)
        train_json['frames'] = train_frames

        with open(os.path.join(os.path.join(front3d_blender_dir, scene_id), f'transforms_train.json'), 'w') as f:
            json.dump(train_json, f, indent=4)

        test_json = {}
        test_json['near'] = float(data['z_nears'])
        test_json['far'] = float(data['z_fars'])
        test_frames = []

        for cam_num, (camera_id, camera) in enumerate(data['camera'].items()):
            if camera_id in train_camera_idx:
                continue
            if 'camera_angle_x' not in test_json:
                width = float(camera['width'])
                focal = float(camera['K'][0][0][0])
                test_json['camera_angle_x'] = float(2 * np.arctan(width/(2*focal)))

            frames_info = {}
            frames_info['file_path'] = os.path.join('test', f"r_{camera_id:04}")
            os.makedirs(os.path.join(front3d_blender_dir, scene_id, 'test'), exist_ok=True)
            Image.fromarray(camera['image']['rgba'].squeeze(0).numpy(), mode='RGBA').save(os.path.join(front3d_blender_dir, scene_id, 'test',
                                                                                                          f"r_{camera_id:04}.png"))
            frames_info['transform_matrix'] = all_poses[cam_num].tolist()
            test_frames.append(frames_info)
        test_json['frames'] = test_frames

        os.makedirs(os.path.join(front3d_blender_dir, scene_id), exist_ok=True)

        with open(os.path.join(os.path.join(front3d_blender_dir, scene_id), f'transforms_test.json'), 'w') as f:
            json.dump(test_json, f, indent=4)




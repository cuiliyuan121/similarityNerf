import blenderproc as bproc
import cv2
import numpy as np
import yaml
import os
import json
import random
import shutil
import debugpy
import sys
import tempfile
import networkx as nx
from tqdm import tqdm
sys.path.append(os.getcwd())
from utils.dataset import check_data, Camera, Scene, sample_connected_component_of_camera_overlap_graph
from utils.prepare_data_utils import shared_argparser, default_output_root, wait_for_gpu
from utils.transform import rotation_mat_dist, WORLD_FRAME_TO_TOTAL3D, CAMERA_FRAME_TO_TOTAL3D
from external import blenderproc_utils
from scipy.spatial.transform import Rotation
import bpy
from mathutils import Matrix
from blenderproc.python.types.MeshObjectUtility import MeshObject
from collections import defaultdict

import bmesh
from mathutils import Vector

class SkipException(Exception):
    pass


def render_3d_front_scene(args):
    # set random seed
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # check data and output directory
    data_dirs = check_data(args.data_dir)

    # load config
    scene_data_config_dir = os.path.join(args.config_dir, 'scene_data.yaml')
    with open(scene_data_config_dir, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # load 3D-FUTURE objects info
    future_model_info_dir = os.path.join(data_dirs['future_dir'], 'model_info.json')
    with open(future_model_info_dir, 'r') as f:
        future_model_info = json.load(f)
    future_model_info = {model['model_id']: model for model in future_model_info}
    # load json file of the scene
    front_scene_json = os.path.join(data_dirs['front_scene_dir'], args.id + '.json')
    with open(front_scene_json, "r") as f:
        front_scene_data = json.load(f)
        # save scene data
    # output_scene_root = default_output_root(args.output_dir, 'room_arrangement_4', args.debug)
    # output_scene_dir = os.path.join(output_scene_root, front_scene_data['uid'])
    # init blenderproc
    if args.gpu_ids != 'all':
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
    bproc.init()
    
    # set renderer
    bproc.renderer.set_cpu_threads(args.cpu_threads)
    bproc.renderer.set_light_bounces(diffuse_bounces=200, glossy_bounces=200, max_bounces=200,
                                     transmission_bounces=200, transparent_max_bounces=200)
    samples = config['renderer']['samples']
    if args.debug:
        samples //= 4
    bproc.renderer.set_max_amount_of_samples(samples)
    bproc.renderer.enable_depth_output(activate_antialiasing=False)

    # set camera intrinsics
    camera_config = config['camera']
    f = camera_config['focal_length']
    height = camera_config['height']
    width = camera_config['width']
    K = np.array([[f, 0, width/2], [0, f, height/2], [0, 0, 1.]])
    if args.debug:
        K[:2] /= 4
        height //= 4
        width //= 4
    bproc.camera.set_intrinsics_from_K_matrix(K, width, height)

    # input_scene_dir = os.path.join(output_scene_dir, 'scene.blend')
    scene1_dir = "/idas/users/cuiliyuan/NeRFSceneUnderstanding/data/blend_new/scene13.blend"
    loaded_objects = bproc.loader.load_blend(scene1_dir, obj_types=['mesh', 'curve', 'hair', 'armature','empty', 'light', 'camera']) # all types in document
    render_data = bproc.renderer.render()
    render_data.update(bproc.renderer.render_segmap(map_by=["instance", "class", "name"]))
   
    # remap instance_segmap to objects
    seg2obj = {}
    obj_id = 0
    name2obj = {obj.get_name(): obj for obj in loaded_objects}
    for instance_attribute_map in render_data['instance_attribute_maps']:
        for inst in instance_attribute_map:
            if inst['name'] not in name2obj:
                continue
            obj = name2obj[inst['name']]
            if not obj.has_cp('uid') or inst['idx'] in seg2obj:
                continue
            seg2obj[inst['idx']] = (obj_id, obj)
            obj_id += 1
    if not seg2obj:
        raise SkipException("No valid mapping from segmentation to objects")

    # segmap data有问题...
    # construction data dict per scene
    data = {'camera': {}, 'object': {}}

    # collect per object data
    uid2jid = {}
    for ele in front_scene_data['furniture']:
        uid2jid[ele['uid']] = ele['jid']
    object_cor_frame = config['object_cor_frame']
    for obj_id, obj in seg2obj.values():
        # set object coordinate frame
        local2world_mat = np.eye(4)
        local2world_mat[:3, :3] = obj.get_rotation_mat()
        local2world_mat = bproc.math.change_source_coordinate_frame_of_transformation_matrix(local2world_mat, object_cor_frame)
        center = obj.get_bound_box().mean(axis=0)
        local2world_mat[:3, 3] = center

        bound_box = obj.get_bound_box(local_coords=True)
        size = np.ptp(bound_box, axis=0) * obj.get_scale()
        size = np.abs(bproc.math.change_coordinate_frame_of_point(size, object_cor_frame))

        uid = obj.get_cp('uid')
        if uid not in uid2jid:
            continue
        jid = uid2jid[uid]
        data['object'][obj_id] = {
            'uid': uid,
            'jid': jid,
            'room_id': obj.get_cp('room_id'),
            'future_category': str(future_model_info[jid]['category']),
            'future_super_category': future_model_info[jid]['super-category'],
            'local2world_mat': local2world_mat.astype(np.float32),
            'size': size.astype(np.float32)
        }

    # collect per camera data
    new_camera_id = 0
    for camera_id in range(len(render_data['colors'])):
        # generate 2D object annotations
        instance_segmap = render_data['instance_segmaps'][camera_id]
        obj2d_annotations = {}
        seg_ids = np.unique(instance_segmap)
        for seg_id in seg_ids:
            if seg_id not in seg2obj:
                continue
            obj_id = seg2obj[seg_id][0]
            obj_mask = instance_segmap == seg_id
            if obj_id not in data['object']:
                continue
            obj2d_annotations[obj_id] = {
                'future_category': data['object'][obj_id]['future_category'],
                'future_super_category': data['object'][obj_id]['future_super_category'],
                'bdb2d': np.array(blenderproc_utils.bbox_from_binary_mask(obj_mask)),
                'segmentation': obj_mask,
                'area': blenderproc_utils.calc_binary_mask_area(obj_mask)
            }

        if not obj2d_annotations:
            continue

        def get_frame_cam2world(frame_id=0):
            bpy.context.scene.frame_set(frame_id)
            cam_location = bpy.context.scene.camera.matrix_world.to_translation()
            cam_rot = bpy.context.scene.camera.matrix_world.to_euler()
            cam_rot = Matrix.Rotation(cam_rot[2], 3, 'Z') @ Matrix.Rotation(cam_rot[1], 3, 'Y') @ Matrix.Rotation(cam_rot[0], 3, 'X')
            cam2world_mat = bproc.math.build_transformation_mat(cam_location, cam_rot)
            return cam2world_mat

        # camera basic info
        data['camera'][new_camera_id] = {
            'cam2world_mat': get_frame_cam2world(new_camera_id).astype(np.float32),
            'K': K.astype(np.float32),
            'height': height,
            'width': width,
            'object': obj2d_annotations,
            'image': {k2: np.array(render_data[k1][camera_id]) for k1, k2 in zip(
                ('colors', 'instance_segmaps', 'depth'), ('color', 'instance_segmap', 'depth'))}
        }
        new_camera_id += 1

    # collect scene data
    data['uid'] = front_scene_data['uid']
    data['seg2obj_id'] = {seg_id: obj_id for seg_id, (obj_id, _) in seg2obj.items()}

    # save scene data
    output_scene_root = default_output_root(args.output_dir, 'scene', args.debug)
    output_scene_dir = os.path.join(output_scene_root, data['uid'])
    # first remove output scene directory to update the data
    shutil.rmtree(output_scene_dir, ignore_errors=True)
    scene = Scene(data)
    scene.save(output_scene_dir)

    # bpy.ops.wm.save_mainfile(filepath="/home/dongwenqi/dl/gen_scene/scene.blend")

def main():
    parser = shared_argparser()
    args = parser.parse_args()

    # Enable debug mode
    if args.debugpy:
        debugpy.listen(5678)
        print('debugpy listening on port 5678')
        debugpy.wait_for_client()

    try:
        render_3d_front_scene(args)
    except SkipException as e:
        print('Skip scene due to:')
        print(e)
        output_scene_root = default_output_root(args.output_dir, 'scene', args.debug)
        skipped_scene_dir = os.path.join(output_scene_root, 'skipped_scenes.txt')
        with open(skipped_scene_dir, 'a') as f:
            f.write(args.id + '\n')


if __name__ == "__main__":
    main()
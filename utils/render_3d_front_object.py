import blenderproc as bproc
from blenderproc.python.utility.MathUtility import MathUtility
import numpy as np
import yaml
import os
import random
import debugpy
import sys
import trimesh
from PIL import Image
import shutil
import json
sys.path.append(os.getcwd())
from utils.dataset import check_data, ObjectNeRF
from utils.prepare_data_utils import shared_argparser, default_output_root, wait_for_gpu
from external import shapenet_renderer_utils


def render_3d_front_object(args):
    # set random seed
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # check data and output directory
    data_dirs = check_data(args.data_dir)

    # load config
    output_root = default_output_root(args.output_dir, 'object', args.debug)
    object_data_config_dir = os.path.join(output_root, 'object_data.yaml')
    with open(object_data_config_dir, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    # load 3D-FUTURE objects info
    future_model_info_dir = os.path.join(data_dirs['future_dir'], 'model_info.json')
    with open(future_model_info_dir, 'r') as f:
        future_model_info = json.load(f)
    future_model_info = {model['model_id']: model for model in future_model_info}
    
    # init blenderproc
    if args.gpu_ids != 'all':
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
    bproc.init()
    
    # set renderer
    bproc.renderer.set_cpu_threads(args.cpu_threads)
    bproc.renderer.set_world_background([1., 1., 1.], 1)
    bproc.renderer.set_light_bounces(diffuse_bounces=200, glossy_bounces=200, max_bounces=200,
                                     transmission_bounces=200, transparent_max_bounces=200)
    bproc.renderer.set_output_format(enable_transparency=True)
    samples = config['renderer']['samples']
    if args.debug:
        samples //= 4
    bproc.renderer.set_max_amount_of_samples(samples)

    # set camera intrinsics
    camera_config = config['camera']
    f = camera_config['focal_length']
    height = camera_config['height']
    width = camera_config['width']
    K = np.array([[f, 0, width/2], [0, f, height/2], [0, 0, 1.]])
    bproc.camera.set_intrinsics_from_K_matrix(K, width, height)
        
    # generate camera poses
    split = args.split
    num_camera = config['num_camera'][split]
    if args.debug:
        num_camera //= 10
    sphere_radius = camera_config['sphere_radius']
    if split in ('val', 'test'):
        camera_locations = shapenet_renderer_utils.get_archimedean_spiral(num_camera, sphere_radius)
    elif split == 'train':
        camera_locations = shapenet_renderer_utils.sample_spherical(num_camera, sphere_radius)
    else:
        raise ValueError(f'Unknown split: {split}')
    camera_poses = shapenet_renderer_utils.look_at(camera_locations, np.zeros((1, 3)))
    for camera_pose in camera_poses:
        bproc.camera.add_camera_pose(camera_pose)
                    
    # load and normalize object with trimesh
    obj_dir = os.path.join(data_dirs['future_dir'], args.id, 'raw_model.obj')
    mesh = trimesh.load(obj_dir)
    object_cor_frame_mat = MathUtility._build_coordinate_frame_changing_transformation_matrix(config['object_cor_frame'])[:3, :3]
    vertices = (object_cor_frame_mat @ mesh.vertices.T).T
    vertices = vertices - (np.min(vertices, axis=0) + np.max(vertices, axis=0)) / 2
    vertices = vertices / np.max(vertices, axis=0) / 2
    mesh.vertices = vertices
    
    # save mesh as obj
    output_object_dir = os.path.join(output_root, args.id)
    obj_folder = os.path.join(output_object_dir, 'obj')
    os.makedirs(obj_folder, exist_ok=True)
    # save with trimesh
    obj_dir = os.path.join(obj_folder, 'normalized_model.obj')
    mesh.export(obj_dir)
    
    # load and render object
    obj = bproc.loader.load_obj(obj_dir)[0]
    obj.set_local2world_mat(np.eye(4))
    
    # wait until GPU has enough memory
    wait_for_gpu(output_root, args.gpu_ids, args.min_gpu_mem, args.render_processes, args.id)
    
    # render the whole pipeline
    render_data = bproc.renderer.render()
    
    # collect per camera data
    camera = {}
    for camera_id, image in enumerate(render_data['colors']):
        # replace background
        background = np.ones_like(image) * 255
        background = Image.fromarray(background)
        image = Image.fromarray(image)
        background.paste(image, mask=image)
        image = background.convert('RGB')

        camera[camera_id] = {
            'cam2world_mat': camera_poses[camera_id].astype(np.float32),
            'K': K.astype(np.float32),
            'height': height,
            'width': width,
            'image': {'color': np.array(image)}
        }
    
    # collect object data
    data = {
        'camera': camera,
        'jid': args.id,
        'future_category': str(future_model_info[args.id]['category']),
        'future_super_category': future_model_info[args.id]['super-category']
    }
    
    # first remove output scene directory to update the data
    shutil.rmtree(output_object_dir, ignore_errors=True)
    # save data
    objnerf = ObjectNeRF(data)
    objnerf.save(output_object_dir)
    
    # remove normalized object if not in debug mode
    if not args.debug:
        shutil.rmtree(obj_folder, ignore_errors=True)


def main():
    parser = shared_argparser()
    parser.add_argument('--split', type=str, default='train', help='Split to render')
    args = parser.parse_args()
    
    # Enable debug mode
    if args.debugpy:
        debugpy.listen(5678)
        print('debugpy listening on port 5678')
        debugpy.wait_for_client()
        
    render_3d_front_object(args)


if __name__ == "__main__":
    main()

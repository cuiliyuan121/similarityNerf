import blenderproc as bproc
import numpy as np
import os
import debugpy
import sys
from PIL import Image
sys.path.append(os.getcwd())
from utils.prepare_data_utils import shared_argparser, default_output_root, wait_for_gpu
from utils.dataset import Scene
from utils.transform import bdb3d_corners

def render_full_image_scene(args):

    background_type = 'checker' # checker or grid
    background_dir = os.path.join('external/blender_templates' , f'{background_type}.blend')
    output_root = default_output_root(args.output_dir, 'scene', args.debug)

    input_scene_dir = os.path.join(args.output_dir, 'scene', args.id)
    output_scene_dir = input_scene_dir
    est_scene = Scene.from_dir(input_scene_dir)

    if args.gpu_ids != 'all':
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids

    # set renderer
    bproc.init()
    bproc.renderer.set_cpu_threads(args.cpu_threads)
    bproc.renderer.set_world_background([1., 1., 1.], 1)
    bproc.renderer.set_light_bounces(diffuse_bounces=1, glossy_bounces=1, max_bounces=1,
                                     transmission_bounces=1, transparent_max_bounces=1)
    bproc.renderer.set_output_format(enable_transparency=True)
    background = bproc.loader.load_blend(background_dir)
    if background_type == 'checker':
        materials = bproc.material.collect_all()
    materials[0].nodes.active.texture_mapping.scale = np.array([50.,50.,50.])

    for idx, (camera_id, camera) in enumerate(est_scene['camera'].items()):
        # init blenderproc
        if idx == 0:
            # set camera intrinsics
            f = camera['K'][0,0]
            height = camera['height']
            width = camera['width']
            K = np.array([[f, 0, width/2], [0, f, height/2], [0, 0, 1.]])
            bproc.camera.set_intrinsics_from_K_matrix(K, width, height)

        camera_pose = camera['cam2world_mat']
        bproc.camera.add_camera_pose(camera_pose)


    pos = background[0].get_location()
    min_z = 1e6
    for object_id, object in est_scene['object'].items():
        obj_z = bdb3d_corners(object).min(0)[-1]
        min_z = min(min_z, obj_z)

    background[0].set_location(location=np.array([0,0, min_z]))

    # wait until GPU has enough memory
    wait_for_gpu(output_root, args.gpu_ids, args.min_gpu_mem, args.render_processes, args.id)
    # render the whole pipeline
    render_data = bproc.renderer.render()

    for camera_id, (est_image_dir, render_image) in enumerate(zip(est_scene['camera'], render_data['colors'])):
        # replace background
        background = np.ones_like(render_image) * 255
        background = Image.fromarray(background)
        render_image = Image.fromarray(render_image)
        background.paste(render_image, mask=render_image)

        est_image = Image.open(os.path.join(input_scene_dir, f'{est_image_dir:04}-rgba.png'))
        est_image = Image.alpha_composite(background, est_image)
        est_image.save(os.path.join(output_scene_dir, str(f'{est_image_dir:04}-full.png')))


def main():
    parser = shared_argparser()
    parser.add_argument('--split', type=str, default='train', help='Split to render')
    args = parser.parse_args()

    # Enable debug mode
    if args.debugpy:
        debugpy.listen(5678)
        print('debugpy listening on port 5678')
        debugpy.wait_for_client()

    render_full_image_scene(args)


if __name__ == "__main__":
    main()

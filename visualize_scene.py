import argparse
import os
import random
import numpy as np
from glob import glob
from tqdm import tqdm
from utils.dataset import check_data, Scene, Camera, Object3D, Object2D
from utils.visualize_utils import SceneVisualizer


def main():
    parser = argparse.ArgumentParser(
        description='Visualize data or results for multi-view scene understanding.')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode.')
    parser.add_argument('--data_dir', type=str, default='data/scene', help='Path to scene directory.')
    parser.add_argument('--id', type=str, default=None, help='Scene id to visualize.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--tasks', nargs='+', type=str, default=[
        'ref_graph', 'scene_vis', 'camera_overlap_graph', 'object_association_graph',
        'bdb3d', 'bdb2d', 'axes', 'segmentation', 'instance_segmap_vis', 'semantic_segmap_vis'
        ], help='Tasks to visualize.')
    parser.add_argument('--cameras', type=int, default=None, help='Number of cameras to visualize.')
    parser.add_argument('--min_object_area', type=float, default=None, help='Minimum object area to visualize, in percentage of the image area.')
    args = parser.parse_args()
    
    # debug mode
    if args.debug:
        args.data_dir += '-debug'
        
    # visualize ref_graph of AutoGetSetDict
    if 'ref_graph' in args.tasks:
        print('Visualizing ref_graph...')
        for c in (Scene, Camera, Object3D, Object2D):
            c.visualize_ref_graph(args.data_dir)
    
    # scan dataset scenes or output results
    if args.id:
        scene_dirs = [os.path.join(args.data_dir, args.id)]
    else:
        print('Scanning scenes...')
        scene_dirs = glob(os.path.join(args.data_dir, '*', 'data.pkl'))
        scene_dirs = [os.path.dirname(d) for d in scene_dirs]
        scene_dirs.sort()
    
    # set random seed
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # visualize scenes
    for scene_dir in tqdm(scene_dirs, desc='Visualizing scenes'):
        # load scene
        tqdm.write(f"Loading scene from {scene_dir}")
        scene = Scene.from_dir(scene_dir)
        if args.min_object_area:
            tqdm.write(f"Filtering objects with area less than {args.min_object_area}% of the image area")
            scene.remove_small_object2d(args.min_object_area)
        if args.cameras:
            tqdm.write(f"Sampling {args.cameras} cameras")
            scene = scene.random_subscene(args.cameras, min_overlap_object=3)
        
        # run visualization
        scene_vis = SceneVisualizer(scene)
        for task in args.tasks:
            if task in ['scene_vis', 'camera_overlap_graph', 'object_association_graph']:
                tqdm.write(f"Visualizing scene: {scene['uid']}, task: {task}")
                getattr(scene_vis, task)(scene_dir)
            if task in ['bdb3d', 'bdb2d', 'axes', 'segmentation', 'instance_segmap_vis', 'semantic_segmap_vis']:
                tqdm.write(f"Visualizing scene: {scene['uid']}, task: {task}")
                getattr(scene_vis, task)()
                scene.save_images(scene_dir, task)


if __name__ == "__main__":
    main()

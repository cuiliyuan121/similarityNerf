import argparse
import os
import random
import numpy as np
from glob import glob
from tqdm import tqdm
from collections import defaultdict
from PIL import Image
from utils.dataset import check_data, Camera, ObjectNeRF
from utils.visualize_utils import ObjectVisualizer, image_grid, image_float_to_uint8


def main():
    parser = argparse.ArgumentParser(
        description='Visualize data for single object NeRF.')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode.')
    parser.add_argument('--data_dir', type=str, default='data/object', help='Path to object directory.')
    parser.add_argument('--id', type=str, default=None, help='object id to visualize.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--tasks', nargs='+', type=str, default=['ref_graph', 'camera', 'category'], help='Tasks to visualize.')
    parser.add_argument('--object_sample_for_category', type=int, default=12, help='The number of object samples for each category.')
    args = parser.parse_args()
    
    # debug mode
    if args.debug:
        args.data_dir += '-debug'
        
    # visualize ref_graph of AutoGetSetDict
    if 'ref_graph' in args.tasks:
        print('Visualizing ref_graph...')
        for c in (Camera, ObjectNeRF):
            c.visualize_ref_graph(args.data_dir)
    
    # scan dataset objects
    if args.id:
        object_dirs = [os.path.join(args.data_dir, args.id)]
    else:
        print('Scanning objects...')
        object_dirs = glob(os.path.join(args.data_dir, '*', 'data.pkl'))
        object_dirs = [os.path.dirname(d) for d in object_dirs]
        object_dirs.sort()
    
    # set random seed
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # visualize differenct categories
    if 'category' in args.tasks:
        categories = defaultdict(list)
        category_nums = defaultdict(lambda: 0)
        random.shuffle(object_dirs)
        
        # collect objects for each category
        for object_dir in tqdm(object_dirs, desc='Scanning categories'):
            objnerf = ObjectNeRF.from_dir(object_dir)
            category = objnerf['category']
            category_nums[category] += 1
            if len(categories[category]) < args.object_sample_for_category:
                categories[category].append(object_dir)
        
        # print category statistics
        print(f"{len(category_nums)} categories found:")
        for category, num in category_nums.items():
            print(f"{category}: {num}")
        
        # visualize objects for each category
        category_vis_dir = os.path.join(args.data_dir, 'category_vis')
        os.makedirs(category_vis_dir, exist_ok=True)
        for category, category_dirs in tqdm(categories.items(), desc='Visualizing categories'):
            tqdm.write(f"Visualizing category '{category}'")
            object_images = []
            for object_dir in category_dirs:
                objnerf = ObjectNeRF.from_dir(object_dir)
                object_image = random.choice(list(objnerf['camera'].values()))['image']['color']
                object_images.append(image_float_to_uint8(object_image))
            category_image = image_grid(object_images, padding=2, background_color=(128, 128, 128), short_height=True)
            category_image_dir = os.path.join(category_vis_dir, f"{category}.png")
            Image.fromarray(category_image).save(category_image_dir)
            
    # visualize objects
    for object_dir in tqdm(object_dirs, desc='Visualizing objects'):
        # load object
        tqdm.write(f"Loading object from {object_dir}")
        objnerf = ObjectNeRF.from_dir(object_dir)
        
        # run visualization
        object_vis = ObjectVisualizer(objnerf)
        for task in args.tasks:
            if task in ['camera']:
                tqdm.write(f"Visualizing object: {objnerf['jid']}, task: {task}")
                getattr(object_vis, task)(object_dir)
            if task in ['bdb3d']:
                tqdm.write(f"Visualizing object: {objnerf['jid']}, task: {task}")
                getattr(object_vis, task)()
                objnerf.save_images(object_dir, task)


if __name__ == "__main__":
    main()

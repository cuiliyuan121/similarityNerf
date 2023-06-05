import mmcv
import os
import numpy as np
import cv2
from glob import glob
import shutil

scene_id = "34ffd30a-32a4-4db0-aeaf-0fc61afec7e0"

path_names = ['output/finetune/all']

scene_dir = os.path.join("/idas/users/cuiliyuan/NeRFSceneUnderstanding/data_debug/scene",scene_id)
pic_dir = glob(os.path.join(scene_dir, '*.png'))
pkl_dir = glob(os.path.join(scene_dir, '*.pkl'))
output_root = '/idas/users/cuiliyuan/NeRFSceneUnderstanding/data_debug/scene-monocular_depth/'
output_pic_dir = os.path.join(output_root, scene_id)

shutil.rmtree(output_pic_dir , ignore_errors=True)
os.makedirs(output_pic_dir, exist_ok=True)

res_path = os.path.join('/idas/users/cuiliyuan/NeRFSceneUnderstanding/data_debug/scene-monocular_depth',scene_id)

weights = [1]

reweights = [w/sum(weights) for w in weights]

file_names = os.listdir(path_names[0])

for name in file_names:
    for idx, (path_name, w) in enumerate(zip(path_names, reweights)):
        file_path = os.path.join(path_name, name)
        if idx == 0:
            temp_res = w * np.load(file_path)
        else:
            temp_res += w * np.load(file_path)
    ensemble_res = temp_res / len(path_names)
    ensemble_res = ensemble_res[0].astype(np.uint16)
    filename = name[:-4].split('-')[0]
    filename = filename + '-depth' + '.png'
    mmcv.imwrite(ensemble_res, os.path.join(res_path, filename))

for pic in pic_dir:
    catagory = ((pic.split('.')[0]).split('/')[-1]).split('-')[-1]
    pic_name = pic.split('/')[-1]
    if catagory == 'instance_segmap' or catagory == 'color':
        os.link(pic, os.path.join(output_pic_dir, pic_name))
for pkl in  pkl_dir:
    pkl_name = pkl.split('/')[-1]
    os.link(pkl, os.path.join(output_pic_dir, pkl_name ))
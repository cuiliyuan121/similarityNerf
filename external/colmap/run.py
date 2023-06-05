import os
import argparse
import subprocess
import json
from tqdm import tqdm
# import pydevd_pycharm
# pydevd_pycharm.settrace('10.186.93.244', port=7771, stdoutToServer=True, stderrToServer=True)

def colmap_est_pose(data_dir, intrinsic, gpu_id):

    cmd = f"colmap feature_extractor \
        --database_path {data_dir}/database.db \
        --image_path {data_dir}/train/ \
        --ImageReader.camera_model PINHOLE \
        --ImageReader.single_camera 1 \
        --SiftExtraction.gpu_index {gpu_id}"

    if intrinsic:
        assert len(intrinsic) == 4
        fx, fy, cx, cy = intrinsic
        cmd += f" --ImageReader.camera_params {fx},{fy},{cx},{cy} "
    subprocess.run(cmd, shell=True, check=True)

    cmd = f"colmap exhaustive_matcher \
        --database_path {data_dir}/database.db \
        --SiftMatching.gpu_index {gpu_id}"
    subprocess.run(cmd, shell=True, check=True)

    os.makedirs(os.path.join(data_dir, 'sparse', '0'), exist_ok=True)
    # os.makedirs(f"{hparams['data_dir']}/sparse/0", exist_ok=True)
    cmd=f"colmap mapper \
            --database_path {data_dir}/database.db \
            --image_path {data_dir}/train \
            --output_path {data_dir}/sparse "
    if intrinsic:
        cmd += ' --Mapper.ba_refine_focal_length 0 '
        cmd += ' --Mapper.ba_refine_extra_params 0 '
    subprocess.run(cmd, shell=True, check=True)

def convert_to_json(root_dir, python_dir, scale_range=False,data_type='nerf'):
    cmd = f"{python_dir} colmap_util/extract_colmap_json.py --root_dir {root_dir} --scale_range True"
    subprocess.run(cmd, shell=True, check=True)

if __name__ == "__main__":

    '''
    command:
    python run.py 
    '''

    colmap_data_dir = 'data/scene-understanding-blender'
    python_dir = '/mnt/data3/dongwenqi/anaconda3/envs/fvor/bin/python'
    colmap_test_list = []
    invalid_list = []
    intrinsic = [519, 519, 320, 240]
    gpu_id = 0
    eval_img_num = 3
    with open(os.path.join(colmap_data_dir, 'test.json'), 'r') as f:
        scene_ids = json.load(f)
    for idx, i in enumerate(tqdm(scene_ids)):
        scene_dir = os.path.join(colmap_data_dir, i)
        colmap_est_pose(scene_dir, intrinsic, gpu_id)
        try:
            convert_to_json(scene_dir, python_dir)
            with open(os.path.join(scene_dir, 'json','transform_info.json'), 'r') as f:
                new_json = json.load(f)
                if eval_img_num == len(new_json.values()):
                    colmap_test_list.append(i)
                else:
                    invalid_list.append(i)

        except:
            print(f"Scene: {i} Could not generate camera pose")
            invalid_list.append(i)

    print(f"Generated total {len(colmap_test_list)} valid scenes")
    print(f"Generated total {len(invalid_list)} invalid scenes")

    with open(os.path.join(colmap_data_dir, "valid.json"), 'w') as f:
        json.dump(colmap_test_list, f, indent=4)

    with open(os.path.join(colmap_data_dir, "invalid.json"), 'w') as f:
        json.dump(invalid_list, f, indent=4)

    print("----------------------------------")
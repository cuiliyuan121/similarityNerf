# similarityNerf

## weights:
   Get it in onedrive cloud.
   
## Command: 

### Evaluate SceneOptimization

```bash
CUDA_VISIBLE_DEVICES=2 python experiment.py --config_dir configs/SceneOptimization.yaml --job test --seed 0 --model.dataset.ids [6eacb9f6-642c-4e02-ae30-4351f4364318]  --model.dataset.max_camera 5 --model.test.log_img_iter 1 ----model.dataset.cameras [[0,1,2,3,4]]
```

### Evaluate SceneUnderstanding

```bash
CUDA_VISIBLE_DEVICES=4 python experiment.py --config_dir configs/SceneUnderstanding.yaml --job test --seed 0 --trainer.limit_test_batches 1 --model.dataset.ids [0c99777a-df50-4c8f-874d-48a1dae04f7c]  --model.dataset.max_camera 1 --model.test.save_results True --model.test.log_img_iter 1
```

### Generate blend file

```bash
python prepare_data.py --job render_3d_front_scenes split_3d_front_scenes calculate_normalization_params --processes 32 --cpu_threads 1 --render_processes 1 --gpu_ids all --id 6eacb9f6-642c-4e02-ae30-4351f4364318
```

### Render 3D-FUTURE with BlenderProc

```bash
blenderproc run utils/render_3d_front_scene_rearrangement.py --data_dir data --output_dir data_debug --config_dir configs --min_gpu_mem 4000 --cpu_threads 1 --render_processes 1 --gpu_ids 1 --id ee842e19-157e-40e6-97aa-5586a153a78e
```

### Generate mask with grounded-segment

```bash
CUDA_VISIBLE_DEVICES=2 python grounded_sam_demo.py --input_image ../data_debug/scene/03cd0e8a-1d94-4228-9d66-e01101830526 --text_prompt "table"
```
The results will be saved as  `output/mask_rcnn.json`.

### Finetune Monocular Depth Estimator

 linke the generated dataset:

```bash
python prepare_data.py --job generate_depth_data --scene_id 34ffd30a-32a4-4db0-aeaf-0fc61afec7e0
```

remeber to change the scene id

After finetuning, generate the depth maps with the following command:

```bash
python tools/test.py configs/depthformer/depthformer_swint_w7_3dfront.py output/finetune/latest.pth --show-dir output/finetune/all --format-only
```

The generated depth maps will be saved in `output/finetune/all`. You'll then need to link the depth maps back to NeRFSceneUnderstanding as a separate dataset:

```bash
python tools/ensemble.py
```

The new dataset will be saved in `data/scene-monocular_depth`.
### **Finetune 2D Detector**

The generated detection results has already been saved as `output/mask_rcnn.json`. You'll then need to convert the detection results to our format by running the following command:

<aside>
💡 remember to change the scene-monocular_depth test.json

</aside>

```bash
python prepare_data.py --job generate_scene_dataset_from_detection_result --scene_id 0c3c3fdb-6f91-466a-8784-17f8f2a12632
```

## scene-understanding

```bash
python prepare_data.py --job generate_scene_understanding_dataset --output_dir data_debug 
```

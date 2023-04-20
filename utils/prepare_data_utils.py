import argparse
import os
import pynvml
import time
import fcntl
import json


def shared_argparser():
    parser = argparse.ArgumentParser(
        description='Prepare data for multi-view scene understanding and object reconstrcution.')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode.')
    parser.add_argument('--debugpy', action='store_true', help='Enable debugpy')
    parser.add_argument('--data_dir', type=str, default='data', help='Input data directory.')
    parser.add_argument('--output_dir', type=str, default='data', help='Output data directory.')
    parser.add_argument('--config_dir', type=str, default='configs', help='Config directory.')
    parser.add_argument('--id', type=str, default=None, help='ID of the scene or object to render.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--min_gpu_mem', type=int, default=4000,
                        help='Minimum GPU memory in MB, will wait until the GPU has at least this amount of memory available')
    parser.add_argument('--cpu_threads', type=int, default=4,
                        help='Number of CPU threads to use for rendering. Best to be set less than the number of CPU cores/render_processes. \
                            Also, some CPU cores should be reserved for preparation processes before rendering. \
                            1 cpu thread means GPU-only rendering')
    parser.add_argument('--render_processes', type=int, default=4,
                        help='Number of processes allowed to render in the same time. Should be set regarding to the memory of the GPU. \
                            if --single_gpu is set, it specifies number of processes allowed to render in the same time on the same GPU.')
    parser.add_argument('--gpu_ids', type=str, default='all', help='GPU ids to use, for example "0,1,2,3", default is "all"')
    return parser


def default_output_root(dir, folder, debug):
    output_dir = os.path.join(dir, folder)
    if debug:
        output_dir += '-debug'
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def initialize_lock(output_dir):
    with open(os.path.join(output_dir, 'render_lock.json'), 'w') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        json.dump({}, f, indent=4)


def wait_for_gpu(output_dir, gpu_ids, min_gpu_mem, render_processes, id, interval=10):
    render_lock_dir = os.path.join(output_dir, 'render_lock.json')
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    time_start = time.time()

    def register_render_process():
        with open(render_lock_dir, 'r+') as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            existing_processes = json.load(f)
            existing_render_processes = existing_processes.get(gpu_ids, [])
            num_existing_render_processes = len(existing_render_processes)
            
            gpu_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            free_memory = gpu_info.free / 1024 // 1024
            
            message = f"GPU memory: {free_memory} MB free/ {min_gpu_mem} MB needed, render processes: {num_existing_render_processes}/{render_processes}"
            if num_existing_render_processes < render_processes and free_memory > min_gpu_mem:
                f.seek(0)
                f.truncate()
                existing_render_processes.append(id)
                existing_processes[gpu_ids] = existing_render_processes
                json.dump(existing_processes, f, indent=4)
                print(f"{message}, start rendering...")
                return True
        print(f"{message}, waiting for GPU..., time elapsed: {time.time() - time_start:.0f}s")
        return False

    while not register_render_process():
        time.sleep(interval)
    
    pynvml.nvmlShutdown()


def release_gpu(output_dir, gpu_ids, id):
    render_lock_dir = os.path.join(output_dir, 'render_lock.json')
    with open(render_lock_dir, 'r+') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        existing_processes = json.load(f)
        existing_gpu_processes = existing_processes.get(gpu_ids, [])
        f.seek(0)
        f.truncate()
        if id in existing_gpu_processes:
            existing_gpu_processes.remove(id)
        json.dump(existing_processes, f, indent=4)

import wandb
import argparse
from glob import glob
from tqdm import tqdm
import shutil
import os
import git


def clean(args):
    print('Cleaning experiments output directories deleted from W&B')
    
    print('Retrieving list of W&B experiments')
    runs = wandb.Api().runs(f"{args.username}/{args.project}")
    exp_ids = [r.id for r in runs]
    
    print('Scanning local experiments')
    exp_dirs = glob(os.path.join(args.output_dir, '*/'))
    for exp_dir in tqdm(exp_dirs, desc='Cleaning'):
        exp_id = os.path.basename(os.path.normpath(exp_dir))
        if exp_id not in exp_ids:
            tqdm.write(f"Deleting {exp_dir}")
            shutil.rmtree(exp_dir, ignore_errors=True)


def clone(args):
    print('Cloning git repo for isolated experiments')
    assert args.suffix is not None, 'Please provide a suffix for the cloned repo'
    
    root_dir = os.path.dirname(os.getcwd())
    dst_folder = os.path.basename(os.getcwd())
    dst_dir = os.path.join(root_dir, f"{dst_folder}-{args.suffix}")
    print(f"Cloning to {dst_dir}")
    repo = git.Repo(os.getcwd())
    repo.clone(path=dst_dir)
    
    for ln_folder in ['data', 'outputs', 'pretrain']:
        print(f"Creating symlink for {ln_folder}")
        src = os.path.join(os.getcwd(), ln_folder)
        dst = os.path.join(dst_dir, ln_folder)
        os.symlink(src, dst)


def main():
    parser = argparse.ArgumentParser('Project utility')
    parser.add_argument('work', type=str, default='clean', help='clean or clone')
    clean_parser = parser.add_argument_group('clean')
    clean_parser.add_argument('--username', type=str, default='pidan1231239', help='W&B username')
    clean_parser.add_argument('--project', type=str, default='nerf_scene_understanding', help='W&B project')
    clean_parser.add_argument('--output_dir', type=str, default='outputs', help='Output directory')
    clone_parser = parser.add_argument_group('clone')
    clone_parser.add_argument('--suffix', type=str, default=None, help='Suffix to add to the cloned git repo')
    args = parser.parse_args()
    
    globals()[args.work](args)


if __name__ == '__main__':
    main()

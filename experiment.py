import wandb
import yaml
import os
import argparse
import torch
import shutil
from glob import glob
from pytorch_lightning import Trainer, seed_everything
from models.scene_optimization import SceneOptimization
from models.auto_encoder_nerf import AutoEncoderNeRF
from models.style_nerf import StyleNeRF
from models.camera_pose_proposal import CameraPoseProposal
from models.object_proposal import ObjectProposal
from models.scene_proposal import SceneProposal
from models.scene_understanding import SceneUnderstanding
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from utils.experiment_utils import read_and_overwrite_config
from utils.torch_utils import MyLearningRateMonitor

# os.environ["WANDB_MODE"] = "offline"


def main():
    parser = argparse.ArgumentParser(
        description="Conduct experiment according to the given configuration.")
    parser.add_argument('--job', type=str, nargs='+', help='Job(s) to run.', default=[], required=True)
    parser.add_argument("--id", type=str, default=None, help="W&B run id for resuming.")
    parser.add_argument("--config_dir", type=str, default=None, help="Path to the configuration file.")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Path to the output directory.")
    parser.add_argument("--project", type=str, default="nerf_scene_understanding", help="W&B project name.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to the checkpoint(s) to load.")
    parser.add_argument("--name", type=str, default=None, help="Name of the run.")
    args, overwrite_args = parser.parse_known_args()

    # create experiment directory
    assert args.id is not None or args.config_dir is not None, "Either id or config_dir must be specified."
    if args.id is None:
        args.id = wandb.util.generate_id()
    exp_dir = os.path.join(args.output_dir, args.id)
    os.makedirs(exp_dir, exist_ok=True)
    print(f"Experiment directory: {exp_dir}")
    if os.environ.get("LOCAL_RANK", None) is None:
        if os.path.islink('latest_vis'):
            os.remove('latest_vis')
        os.symlink(os.path.join(exp_dir, 'wandb', 'latest-run', 'files', 'media'), 'latest_vis')

    # choose config file
    if args.config_dir is None:
        config_dir = glob(os.path.join(exp_dir, "*.yaml"))
        assert len(config_dir) <= 1, "More than one config file found."
        args.config_dir = config_dir[0]
    print(f"Using config file: {args.config_dir}")
    config = read_and_overwrite_config(args.config_dir, overwrite_args)

    # initialize wandb logger
    print("Initializing wandb")
    log_config = config.copy()
    log_config['args'] = vars(args)
    wandb_logger = WandbLogger(project=args.project, id=args.id, config=log_config,
                               save_dir=exp_dir, name=args.name)

    # in the main process, save the config file
    config_name = os.path.basename(args.config_dir)
    if os.environ.get("LOCAL_RANK", None) is None:
        # save config into experiment folder as a yaml file
        print(f"Saving config to {wandb.run.dir}")
        output_config_dir = os.path.join(wandb.run.dir, config_name)
        with open(output_config_dir, 'w') as f:
            yaml.dump(config, f)
        # copy config file to wandb directory
        shutil.copyfile(output_config_dir, os.path.join(exp_dir, config_name))

    # set seed
    print(f"Setting seed to {config['seed']}")
    seed_everything(config['seed'], workers=True)

    # choose checkpoint
    print("Checking which checkpoint to load")
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    Model = globals()[os.path.splitext(config_name)[0].split('-')[0]]
    if args.checkpoint is None:
        last_checkpoint = os.path.join(ckpt_dir, "last.ckpt")
        if os.path.exists(last_checkpoint):
            args.checkpoint = last_checkpoint
            print(f"No checkpoint provided, but found {args.checkpoint}, resuming from it.")
            model = Model.load_from_checkpoint(args.checkpoint, **config['model'])
            print("No sanity check will be performed.")
            config['trainer']['num_sanity_val_steps'] = 0
        else:
            print(f"No checkpoint provided, and no checkpoint found in {ckpt_dir}, starting from scratch")
            model = Model(**config['model'])
    else:
        print(f"Using provided checkpoint {args.checkpoint}")
        model = Model.load_from_checkpoint(args.checkpoint, **config['model'])
        args.checkpoint = None

    # prepare trainer
    # log gradients and model topology
    wandb_logger.watch(model, log_freq=1000)
    # learning rate monitor
    lr_monitor = MyLearningRateMonitor(logging_interval='epoch')
    # checkpoint callback
    checkpoint_callback = ModelCheckpoint(dirpath=ckpt_dir, save_last=True, every_n_epochs=1)
    # construct trainer
    trainer = Trainer(logger=wandb_logger, callbacks=[checkpoint_callback, lr_monitor], **config['trainer'])
        
    if 'train' in args.job:
        print("Training model")
        trainer.fit(model=model, ckpt_path=args.checkpoint)
        args.checkpoint = None  # reset checkpoint to None to avoid loading it again when testing
        # TODO: use the best checkpoint instead of the latest weight
    
    if 'test' in args.job:
        print("Testing model")
        trainer.test(model=model)

    if 'predict' in args.job:
        print("Predicting")
        trainer.predict(model=model)


if __name__ == "__main__":
    main()

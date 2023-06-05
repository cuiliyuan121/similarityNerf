import yaml
from .general_utils import recursive_merge_dict


def read_and_overwrite_config(config_dir, overwrite_args):
    # read config file
    print(f"Reading config file from {config_dir}")
    with open(config_dir, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # parse overwrite args
    print(f"Parsing overwrite args: {overwrite_args}")
    overwrite_config = {}
    for overwrite_key, overwrite_value in zip(overwrite_args[::2], overwrite_args[1::2]):
        overwrite_key = overwrite_key.replace("--", "").split('.')
        overwrite_value = yaml.load(overwrite_value, Loader=yaml.FullLoader)
        if len(overwrite_key) > 1:
            overwrite_dict = overwrite_config
            for key in overwrite_key[:-1]:
                if key not in overwrite_dict:
                    overwrite_dict[key] = {}
                overwrite_dict = overwrite_dict[key]
        else:
            overwrite_dict = overwrite_config
        overwrite_dict[overwrite_key[-1]] = overwrite_value

    print("Merging config and overwrite_config")
    config = recursive_merge_dict(config, overwrite_config)
    
    return config

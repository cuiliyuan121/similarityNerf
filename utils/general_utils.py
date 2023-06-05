def recursive_merge_dict(target_dict, overwrite_dict):
    new_dict = target_dict.copy()
    for key, value in overwrite_dict.items():
        if isinstance(value, dict) and key in target_dict:
            new_dict[key] = recursive_merge_dict(target_dict[key], value)
        else:
            new_dict[key] = value
    return new_dict


def recursive_getattr(obj, attr_name):
    attr_names = attr_name.split('.')
    attr = obj
    for a in attr_names:
        attr = getattr(attr, a)
    return attr

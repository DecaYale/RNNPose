
from easydict import EasyDict as edict
import os
import shutil
import yaml


def mkdir_if_not_exists(path):
    """Make a directory if it does not exist.
    Args:
        path: directory to create
    """
    if not os.path.exists(path):
        os.makedirs(path)


def read_yaml(filename):
    """Load yaml file as a dictionary item
    Args:
        filename (str): .yaml file path
    Returns:
        cfg (dict): configuration
    """
    if filename is not None:
        with open(filename, 'r') as f:
            return yaml.load(f, Loader=yaml.FullLoader)
    else:
        return {}


def copy_file(src_file, tgt_file):
    """Copy a file
    Args:
        src_file (str): source file
        tgt_file (str): target file
    """
    shutil.copyfile(src_file, tgt_file)


def update_dict(dict1, dict2, intersection=False):
    """update dict1 according to dict2
    Args:
        dict1 (dict): reference dictionary
        dict2 (dict): new dictionary
    return
        dict1 (dict): updated reference dictionary
    """
    for item in dict2:
        # if dict1.get(item, -1) != -1:
        if item in dict1:
            if isinstance(dict1[item], dict):
                dict1[item] = update_dict(dict1[item], dict2[item], intersection)
            else:
                dict1[item] = dict2[item]
        else:
            if not intersection:
                dict1[item] = dict2[item]
            else:
                raise ValueError(f"Key '{item}' is in the second dict but not in the first dict!")
    #inverse check
    for item in dict1:
        if item not in dict2:
            print(f"Warning: key {item} is not given and will use the default values")
    
    return dict1


def merge_cfg(cfg_files, intersection=False):
    """merge default configuration and custom configuration
    Args:
        cfg_files (str): configuration file paths [default, custom]
    Returns:
        cfg (edict): merged EasyDict
    """
    edict_items = []
    # cfg = {}
    cfg = read_yaml(cfg_files[0])
    # for f in cfg_files:
    for f in cfg_files[1:]:
        # if f is not None:
        if os.path.exists(f):
            cfg = update_dict(cfg, read_yaml(f), intersection)
        else:
            raise ValueError(f"File {f} does not exist." )
    return edict(cfg)


def write_cfg(default, custom, f, level_cnt=0):
    """write configuration to file
    Args:
        default (dict): default configuration dictionary
        custom (dict): custom configuration dictionary
        file (TextIOWrapper)
    """
    offset_len = 100
    for item in default:
        if isinstance(default[item], dict):
            if custom.get(item, -1) == -1:
                custom[item] = {}
            line = "  "*level_cnt + item + ": "
            offset = offset_len - len(line)
            line += " "*offset + " # |"
            f.writelines(line + "\n")
            write_cfg(default[item], custom[item], f, level_cnt+1)
        else:
            line = "  " * level_cnt + item + ": "
            if custom.get(item, -1) == -1:
                if default[item] is not None:
                    line += str(default[item])
                offset = offset_len - len(line)
                line += " "*offset + " # | "
            else: 
                if custom[item] is not None:
                    line += str(custom[item])
                offset = offset_len - len(line)
                line += " "*offset + " # | "
                if custom[item] != default[item]:
                    line += str(default[item])
            f.writelines(line)
            f.writelines("\n")


def save_cfg(cfg_files, file_path):
    """Save configuration file
    Args:
        cfg_files (str): configuration file paths [default, custom]
    Returns:
        cfg (edict): merged EasyDict
    """
    # read configurations
    default = read_yaml(cfg_files[0])
    custom = read_yaml(cfg_files[1])

    # create file to be written
    f = open(file_path, 'w')

    # write header line
    line = "# " + "-"*20 + " Setup " + "-"*74
    line += "|" + "-"*10 + " Default " + "-"*20 + "\n"
    f.writelines(line)

    # write configurations
    write_cfg(default, custom, f)
    f.close()

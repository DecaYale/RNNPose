from yacs.config import CfgNode as CN
from utils.singleton import Singleton
import os

def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    # if type(a) is not dict:
    if not isinstance(a, (dict)):
        return

    for k, v in a.items():
        # a must specify keys that are in b
        if not k in b:
            raise KeyError('{} is not a valid config key'.format(k))

        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(('Type mismatch ({} vs. {}) '
                                'for config key: {}').format(type(b[k]),
                                                            type(v), k))

        # recursively merge dicts
        # if type(v) is dict:
        if isinstance(v, (dict)):
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print('Error under config key: {}'.format(k))
                raise
        else:
            b[k] = v



class Config(metaclass=Singleton):
    def __init__(self):
        ##############  ↓  Basic   ↓  ##############
        self.ROOT_CN = CN()
        self.ROOT_CN.BASIC = CN()
        self.ROOT_CN.BASIC.input_size=[480,640] #h,w
        self.ROOT_CN.BASIC.crop_size=[320,320] #h,w
        self.ROOT_CN.BASIC.zoom_crop_size=[320,320] #h,w
        self.ROOT_CN.BASIC.render_image_size=[320,320]#h,w
        self.ROOT_CN.BASIC.patch_num=64#h,w

        ##############  ↓  LM OPTIM   ↓  ##############
        self.ROOT_CN.LM=CN()
        self.ROOT_CN.LM.LM_LMBDA= 0.0001
        self.ROOT_CN.LM.EP_LMBDA=100

        ##############  ↓  data   ↓  ##############
        self.ROOT_CN.DATA=CN()
        self.ROOT_CN.DATA.OBJ_ROOT="" #h,w
        self.ROOT_CN.DATA.VOC_ROOT=f"{os.path.dirname(os.path.abspath(__file__)) }/../EXPDATA/" #h,w

    def __get_item__(self, key):
        return self.ROOT_CN.__getitem__(key)
    
    def merge(self, config_dict, sub_key=None):

        if sub_key is not None:
            _merge_a_into_b(config_dict, self.ROOT_CN[sub_key])
        else:
            _merge_a_into_b(config_dict, self.ROOT_CN)

##############  ↓  Model  ↓  ##############
# _CN.model = CN()
# _CN.model.input_size=[480,640]
# _CN.model.crop_size=[320,320] 
# def get_cfg_defaults():

def get_cfg(Node=None):
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    # return _CN.clone()
    if Node is None:
        return  Config()
    else:
        return Config().__get_item__(Node)

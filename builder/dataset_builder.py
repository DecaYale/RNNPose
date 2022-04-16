
from data.dataset import get_dataset_class
import numpy as np
from functools import partial
from data.preprocess import preprocess, preprocess_deepim, patch_crop


def build(input_reader_config,
          training,
          ):

    prep_cfg = input_reader_config.preprocess
    dataset_cfg = input_reader_config.dataset
    cfg = input_reader_config
    
    dataset_cls = get_dataset_class(dataset_cfg.dataset_class_name)

    if 0:# 'DeepIM' in dataset_cfg.dataset_class_name:
        # patch_cropper = partial(patch_crop, margin_ratio=0.2, output_size=256 )
        patch_cropper = None 
        prep_func = partial(preprocess_deepim, 
                        max_points=dataset_cfg.max_points,
                        correspondence_radius=prep_cfg.correspondence_radius_threshold,
                        patch_cropper=patch_cropper,
                        image_scale=prep_cfg.get('image_scale', 1),
        ) 

    else:
        prep_func = partial(preprocess, 
                        max_points=dataset_cfg.max_points,
                        correspondence_radius=prep_cfg.correspondence_radius_threshold,
                        image_scale=prep_cfg.get('image_scale', 1),
                        crop_param=prep_cfg.get('crop_param', None),
                        kp_3d_param=prep_cfg.get('kp_3d_param', {"kp_num":30} ),
                        use_coords_as_3d_feat=prep_cfg.get('use_coords_as_3d_feat', False)
        ) 
   

    dataset = dataset_cls(
        info_path=dataset_cfg.info_path,
        root_path=dataset_cfg.root_path,
        model_point_dim=dataset_cfg.model_point_dim,
        is_train=training,
        prep_func=prep_func,
        seq_names=dataset_cfg.get('seq_names', None),
        cfg=dataset_cfg
    )

    return dataset

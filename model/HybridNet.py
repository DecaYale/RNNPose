
import torch 
import torch.nn as nn 

from thirdparty.kpconv.kpconv_blocks import *
import torch.nn.functional as F
import numpy as np
from kpconv.lib.utils import square_distance
from model.descriptor2D import  SuperPoint2D
from model.descriptor3D import  KPSuperpoint3Dv2



REGISTERED_HYBRID_NET_CLASSES={}
def register_hybrid_net(cls, name=None):
    global REGISTERED_HYBRID_NET_CLASSES
    if name is None:
        name = cls.__name__
    assert name not in REGISTERED_HYBRID_NET_CLASSES, f"exist class: {REGISTERED_HYBRID_NET_CLASSES}"
    REGISTERED_HYBRID_NET_CLASSES[name] = cls
    return cls


def get_hybrid_net(name):
    global REGISTERED_HYBRID_NET_CLASSES
    assert name in REGISTERED_HYBRID_NET_CLASSES, f"available class: {REGISTERED_HYBRID_NET_CLASSES}"
    return REGISTERED_HYBRID_NET_CLASSES[name]

class ContextFeatureNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.context_fea_extractor_3d= KPSuperpoint3Dv2(config['context_fea_extractor_3d'] )
    
    def forward(self, batch):
        # x = batch['features'].clone().detach()
        # assert len(batch['stack_lengths'][-1])==1, "Only support bs=1 for now" 
        len_src_c = batch['stack_lengths'][-1][0]
        pcd_c = batch['model_points'][-1]
        pcd_c = pcd_c[:len_src_c]

        image=batch['image']

        ############### encode 3d and 2d features ###############
        batch3d={
            'points': batch['model_points'], 
            'neighbors': batch['neighbors'], 
            'pools':  batch['pools'], 
            'upsamples': batch['upsamples'],
            'features': batch['model_point_features'], 
            'stack_lengths': batch['stack_lengths'],
        }
        ctx_descriptors_3d = self.context_fea_extractor_3d(batch3d)


        return {
            "ctx_fea_3d":ctx_descriptors_3d,
        }



@register_hybrid_net
class HybridDescNet(nn.Module):
    #independent 2d and 3d network
    def __init__(self, config):
        super().__init__()

        self.corr_fea_extractor_2d= SuperPoint2D(config['keypoints_detector_2d'] )
        self.corr_fea_extractor_3d= KPSuperpoint3Dv2(config['keypoints_detector_3d'] )
        self.descriptors_3d = {}


    def forward(self, batch):
        assert len(set(batch['class_name']))==1, "A batch should contain data of the same class."
        class_name = batch['class_name'][0]

        len_src_c = batch['stack_lengths'][-1][0]
        pcd_c = batch['model_points'][-1]
        pcd_c = pcd_c[:len_src_c]#, pcd_c[len_src_c:]

        image=batch['image']

        ############### encode 3d and 2d features ###############
        batch3d={
            'points': batch['model_points'], 
            'neighbors': batch['neighbors'], 
            'pools':  batch['pools'], 
            'upsamples': batch['upsamples'],
            'features': batch['model_point_features'], 
            'stack_lengths': batch['stack_lengths'],
        }
        if self.training:
            self.descriptors_3d[class_name] = self.corr_fea_extractor_3d(batch3d)
        else:
            if class_name not in self.descriptors_3d:
                self.descriptors_3d[class_name] = self.corr_fea_extractor_3d(batch3d)

        descriptors_2d = self.corr_fea_extractor_2d(image)['descriptors']


        return {
            "descriptors_2d":descriptors_2d,
            "descriptors_3d":self.descriptors_3d[class_name],
            "scores_saliency_3d": None, 
            "scores_overlap_3d":None, 

        }

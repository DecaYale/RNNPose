#
import time
import torch
from torch import nn
from torch.nn import functional as F
import torch.distributed as dist
import apex
import numpy as np
import os
from easydict import EasyDict as edict
from transforms3d.euler import mat2euler, euler2mat, euler2quat, quat2euler
from functools import partial



from model.HybridNet import HybridDescNet,ContextFeatureNet
from thirdparty.kpconv.lib.utils import square_distance
from model.PoseRefiner import PoseRefiner  

from utils.pose_utils import pose_padding
from geometry.transformation import SE3Sequence

from geometry.diff_render_optim import DiffRendererWrapper
from config.default import get_cfg
from utils.util import dict_recursive_op

# from model.RNNPose import register_posenet

REGISTERED_NETWORK_CLASSES = {}


def register_posenet(cls, name=None):
    global REGISTERED_NETWORK_CLASSES
    if name is None:
        name = cls.__name__
    assert name not in REGISTERED_NETWORK_CLASSES, f"exist class: {REGISTERED_NETWORK_CLASSES}"
    REGISTERED_NETWORK_CLASSES[name] = cls
    return cls


def get_posenet_class(name):
    global REGISTERED_NETWORK_CLASSES
    assert name in REGISTERED_NETWORK_CLASSES, f"available class: {REGISTERED_NETWORK_CLASSES}"
    return REGISTERED_NETWORK_CLASSES[name]




@register_posenet
class RNNPose(nn.Module):
    def __init__(self,
                 criterions,
                 opt,
                 name="RNNPose",
                 **kwargs):
        super().__init__()

        self.name = name
        self.opt = opt

        self.hybrid_desc_net=HybridDescNet(opt.descriptor_net)
        self.ctx_fea_net = ContextFeatureNet(opt.descriptor_net)

        self.ctx_fea = {} 

        self.render_params = edict({
            "width": opt.input_w,  # 128,
            "height": opt.input_h,  # 128,
            "gpu_id": opt.get('gpu_id', 0),
            "obj_seqs": opt.obj_seqs
        })

        renderer, diff_renderer= self._render_init(self.render_params)
        self.diff_renderer = diff_renderer

        self.motion_net = PoseRefiner(
            opt.motion_net, bn_is_training=self.training, is_training=self.training,
            renderer=diff_renderer 
        )
        self.contrastive_loss = criterions.get(
            "metric_loss", None)
        self.pose_loss = criterions.get("pose_loss", None)

        self.register_buffer("global_step", torch.LongTensor(1).zero_())


    def update_global_step(self):
        self.global_step += 1

    def get_global_step(self):
        return int(self.global_step.cpu().numpy()[0])

    def clear_global_step(self):
        self.global_step.zero_()
    
    def sample_poses(self, pose_tgt):
        SYN_STD_ROTATION = 15
        SYN_STD_TRANSLATION = 0.01
        ANGLE_MAX=45
        pose_src = pose_tgt.copy()
        num = pose_tgt.shape[0]
        for i in range(num):
            euler = mat2euler(pose_tgt[i, :3, :3])
            euler += SYN_STD_ROTATION * np.random.randn(3) * np.pi / 180.0
            pose_src[i, :3, :3] = euler2mat(euler[0], euler[1], euler[2])

            pose_src[i, 0, 3] = pose_tgt[i, 0, 3]+ SYN_STD_TRANSLATION * np.random.randn(1)
            pose_src[i, 1, 3] = pose_tgt[i, 1, 3] + SYN_STD_TRANSLATION * np.random.randn(1)
            pose_src[i, 2, 3] = pose_tgt[i, 2, 3]  + 5 * SYN_STD_TRANSLATION * np.random.randn(1)

            r_dist = np.arccos((np.trace(pose_src[i, :3,:3].transpose(-1,-2) @ pose_tgt[i, :3,:3]) - 1 )/2)/np.pi*180

            while r_dist > ANGLE_MAX:#or not (16 < center_x < (640 - 16) and 16 < center_y < (480 - 16)):
                print("r_dist > ANGLE_MAX, resampling...")
                euler = mat2euler(pose_tgt[i, :3, :3])
                euler += SYN_STD_ROTATION * np.random.randn(3) * np.pi / 180.0
                pose_src[i, :3, :3] = euler2mat(euler[0], euler[1], euler[2])

                pose_src[i, 0, 3] = pose_tgt[i, 0, 3]+ SYN_STD_TRANSLATION * np.random.randn(1)
                pose_src[i, 1, 3] = pose_tgt[i, 1, 3] + SYN_STD_TRANSLATION * np.random.randn(1)
                pose_src[i, 2, 3] = pose_tgt[i, 2, 3]  + 5 * SYN_STD_TRANSLATION * np.random.randn(1)

                r_dist = np.arccos((np.trace(pose_src[i, :3,:3].transpose(-1,-2) @ pose_tgt[i, :3,:3]) - 1 )/2)*np.pi/180
        return pose_src

    def _render_init(self, config):
        # from data.ycb.basic import bop_ycb_class2idx
        print("config.gpu_id:", config.gpu_id)

        obj_paths = []
        tex_paths = []

        # build cls2idx table for the renderer
        cls2idx = {}
        LM_SEQ=["ape", "benchvise", "camera","cam", "can", "cat", "driller", "duck", "eggbox", "glue", "holepuncher", "iron", "lamp", "phone"]
        # YCB_SEQ=bop_ycb_class2idx.keys()
        for i, seq in enumerate(set(config.obj_seqs)):

            if seq in LM_SEQ: 
                obj_path = f'{os.path.dirname(__file__)}/../EXPDATA/LM6d_converted/models/{seq}/textured.obj'
                tex_path = f'{os.path.dirname(__file__)}/../EXPDATA/LM6d_converted/models/{seq}/texture_map.png'
                assert os.path.exists(obj_path), f"'{obj_path}' dose not exist!" 
                assert os.path.exists(tex_path), f"'{tex_path}' dose not exist!" 
                obj_paths.append(obj_path)
                tex_paths.append(tex_path)
                cls2idx[seq] = i
            else:
                raise NotImplementedError
        renderer=None

        diff_renderer = DiffRendererWrapper(obj_paths)
        diff_renderer.cls2idx = cls2idx

        return renderer, diff_renderer


    def forward(self, sample):
        assert len(set(sample['class_name']))==1, "A batch should contain data of the same class."
        class_name = sample['class_name'][0]

        #encode 3d-2d descriptors
        preds_dict=self.hybrid_desc_net(sample)

        len_src_f = sample['stack_lengths'][0][0]
        geofea_3d = preds_dict.get('descriptors_3d', None) 
        geofea_2d = preds_dict.get('descriptors_2d', None) 


        #encode 3D context features 
        if self.training:
            self.ctx_fea[class_name]=self.ctx_fea_net(sample)
        else:
            if class_name not in self.ctx_fea:
                self.ctx_fea[class_name]=self.ctx_fea_net(sample)

        preds_dict.update(self.ctx_fea[class_name])
        ctx_fea_3d = preds_dict['ctx_fea_3d'][:len_src_f]

        if self.training:
            pose=pose_padding(sample['original_RT'])

            if sample.get("rendered_RT", None) is not None:
                syn_pose = pose_padding(sample['rendered_RT'])
            else:
                syn_pose = torch.tensor(self.sample_poses(sample['original_RT'].detach().cpu(
                ).numpy()), device=sample['original_RT'].device, dtype=sample['original_RT'].dtype)
                syn_pose = pose_padding(syn_pose)
        else:
            pose = pose_padding(sample['original_RT'])
            syn_pose = pose_padding(sample['rendered_RT'])
            

        # calculate the GT relative pose and the initial pose
        
        Ts_pred = SE3Sequence(
            matrix=torch.stack([syn_pose ], dim=1))
        mot_res = self.motion_net(
            Ts=Ts_pred,  
            intrinsics=sample['K'],
            image=sample['image'], 
            fea_3d=ctx_fea_3d[None], 
            Tj_gt=SE3Sequence(matrix=pose[:, None]),
            obj_cls=sample['class_name'],
            geofea_2d = geofea_2d, 
            geofea_3d = geofea_3d[None]
        )
        preds_dict.update(mot_res)
        sample['syn_depth'] = mot_res['syn_depth']
        if self.training:
            ret = self.loss(sample, preds_dict)

            ret['syn_img'] = mot_res['syn_img']
            ret['syn_depth'] = mot_res['syn_depth']
            ret['flow'] = mot_res['flow']
            ret['weight'] = mot_res['weight']

            return ret
        else:
            # ret = self.loss(sample, preds_dict)
            ret={}
            ret.update(preds_dict)
            return ret

    
    def loss(self, sample, preds_dict):

        len_src_f = sample['stack_lengths'][0][0]
        RT =sample['RT']
        camera_intrinsic=sample['K']
        descriptors_2d_map = preds_dict['descriptors_2d']
        descriptors_3d = preds_dict['descriptors_3d'][:len_src_f]
        rand_descriptors_3d = preds_dict['descriptors_3d'][len_src_f:]
        model_points=sample['model_points'][0][:len_src_f]
        orig_model_points = sample['original_model_points']
        rand_model_points=sample['model_points'][0][len_src_f:]
        lifted_points=sample['lifted_points'][0].squeeze(0)
        correspondence=sample['correspondences_2d3d'].squeeze(0)
        depth = sample['depth']
        

        # get the foreground 2d descriptors 
        ys_, xs_=torch.nonzero(sample['depth'].squeeze(), as_tuple=True )
        descriptors_2d=descriptors_2d_map[:,:,ys_,xs_].squeeze().permute([1,0])

        if self.training: 
            device= lifted_points.device
            fg_point_num = len(lifted_points)
            model_point_num = len(model_points)
          
            # append bg features 
            ys_bg, xs_bg = torch.nonzero(sample['depth'].squeeze()<=0, as_tuple=True )
            descriptors_2d_bg = descriptors_2d_map[:,:,ys_bg, xs_bg].squeeze().permute([1,0])
            # descriptors_2d_bg = descriptors_2d_bg[np.random.randint(0,len(descriptors_2d_bg), size=len(lifted_points) )]
            descriptors_2d = torch.cat([descriptors_2d, descriptors_2d_bg], dim=0) 
            descriptors_3d = torch.cat([descriptors_3d, descriptors_2d_bg], dim=0 )

            # append handcrafted coordinates to simplify the code(assign the same coordinates for the bg points far away from the fg points, i.e. 10e6 )
            lifted_points = torch.cat([lifted_points, torch.ones([len(descriptors_2d_bg) ,3],device=device )*10e6 ] ) #append very distant points
            model_points = torch.cat([model_points, torch.ones([len(descriptors_2d_bg) ,3], device=device)*10e6 ] ) #append very distant points

            #append one-to-one inds
            if len(descriptors_2d_bg)>0:
                # randomly sample the bg points to balance the learning process
                sample_inds=np.random.randint(0,len(descriptors_2d_bg), size=int(len(correspondence)*0.1) )
           
                bg_corr= torch.stack([
                    torch.arange(fg_point_num, fg_point_num+len(descriptors_2d_bg), device=device )[sample_inds], 
                    torch.arange(model_point_num, model_point_num+len(descriptors_2d_bg), device=device)[sample_inds]
                 ], dim=-1) #Nx2
                correspondence = torch.cat([correspondence, bg_corr ], dim=0)
            
        if len(lifted_points)>0:
            contra_loss=self.contrastive_loss(src_pcd=lifted_points, tgt_pcd=model_points,
                            src_feats=descriptors_2d, tgt_feats=descriptors_3d,
                            correspondence=correspondence,
                            scores_overlap=None, scores_saliency=None)
        else:
            print("Warning: Contrastive loss is skipped, as no lifted point is found!", flush=True)
            contra_loss={
                'circle_loss': torch.zeros([1], device=depth.device),
                'recall': torch.zeros([1], device=depth.device)
            }


        loss3d = partial(self.pose_loss.forward,
                         points=orig_model_points[None])
        motion_loss = self.motion_net.compute_loss(
            preds_dict['Tij_gt'], sample['syn_depth'], intrinsics=sample['K'], loss='l1', log_error=True, loss3d=loss3d)

        
        loss =  self.contrastive_loss.weight * contra_loss['circle_loss']  + motion_loss['total_loss'] 

        res = {
            "loss": loss, 
            "circle_loss": contra_loss['circle_loss'].detach(),
            "recall":contra_loss['recall'].detach(),
            # "geometric_loss": torch.zeros(1),
            "reproj_loss": motion_loss['reproj_loss'].detach(),
            "loss_3d_proj": motion_loss['loss_3d_proj'].detach(), 
            "valid_mask": motion_loss["valid_mask"].detach(),
        }
        return res


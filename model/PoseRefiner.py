import os 
import time
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
from easydict import EasyDict as edict
from functools import partial

from geometry.transformation import *
from geometry.intrinsics import *
from geometry.projective_ops import coords_grid, normalize_coords_grid
from model.CFNet import GRU_CFUpdator , ImageFeaEncoder
from utils.pose_utils import pose_padding
from config.default import get_cfg



EPS = 1e-5
MIN_DEPTH = 0.1
MAX_ERROR = 100.0

# exclude extremly large displacements
MAX_FLOW = 400


def raft_sequence_flow_loss(flow_preds, flow_gt, valid, gamma=0.8, max_flow=MAX_FLOW):
    """ Loss function defined over sequence of flow predictions """

    n_predictions = len(flow_preds)    
    flow_loss = 0.0
    
    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt**2, dim=1).sqrt()
    
    valid = (valid >= 0.5) & (mag < max_flow)

    for i in range(n_predictions):
        i_weight = gamma**(n_predictions - i - 1)
        i_loss = (flow_preds[i] - flow_gt).abs()
        flow_loss += i_weight * (valid[:, None] * i_loss).mean()

    epe = torch.sum((flow_preds[-1] - flow_gt)**2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
    }

    return flow_loss, metrics




class PoseRefiner(nn.Module):
    def __init__(self, cfg,
                 reuse=False,
                 schedule=None,
                 use_regressor=True,
                 is_calibrated=True,
                 bn_is_training=False,
                 is_training=True,
                 renderer=None,
                 ):

        super().__init__()

        self.legacy=True
        self.cfg = cfg
        self.reuse = reuse
        self.sigma=nn.ParameterList( [nn.Parameter(torch.ones(1)*1 )] )
        self.with_corr_weight = self.cfg.get("with_corr_weight", True)
        if not self.with_corr_weight:
            print("Warning: the correlation weighting is disabled.")

        self.is_calibrated = cfg.IS_CALIBRATED
        if not is_calibrated:
            self.is_calibrated = is_calibrated

        self.is_training = is_training
        self.use_regressor = use_regressor

        self.residual_pose_history = []
        self.Ti_history =[]
        self.coords_history = []
        self.residual_history = []
        self.inds_history = []
        # self.weights_history = []
        self.flow_history = []
        self.intrinsics_history = []
        self.summaries = []

        if self.cfg.FLOW_NET=='raft':
            self.image_fea_enc= ImageFeaEncoder()
            self.cf_net = GRU_CFUpdator(self.cfg.raft) 
        else:
            raise NotImplementedError 
        self.renderer = renderer  

    def _clear(self,):
        self.residual_pose_history = []
        self.Ti_history =[]
        self.coords_history = []
        self.residual_history = []
        self.inds_history = []
        # self.weights_history = []
        self.flow_history = []
        self.intrinsics_history = []
        self.summaries = []

    def __len__(self):
        return len(self.residual_pose_history)

    def render(self, params, render_tex=False):
        """ render a batch of images given the intrinsic and extrinsic params 

        Args:
            params.K: np.array, of shape Bx3x3
            params.camera_extrinsics: np.array, of shape Bx3x4

        Returns:
            [type]: [description]
        """

        bs=params.K.shape[0]
        colors=[]
        depths=[]

        color,depth= self.renderer( params.obj_cls, params.vert_attribute, T=params.camera_extrinsics, K=params.K, 
            render_image_size=params.render_image_size, near=0.1, far=6, render_tex=render_tex)
        #set the invalid values to zeros
        depth[depth==-1] = 0
        return {
            # 1x3xHxW
            "syn_img": color, 
            "syn_depth": depth.detach(), # 1x1xHxW
        }


    def get_affine_transformation(self, mask, crop_center, with_intrinsic_transform=False, output_size=None, margin_ratio=0.4):
        B,_,H,W = mask.shape
        ratio = float(H) / float(W)
        affine_matrices = []
        intrinsic_matrices = []

        for b in range(B):
            zoom_c_x, zoom_c_y = crop_center[b] #crop_center

            ys, xs = np.nonzero(mask[b][0].detach().cpu().numpy() )
            if len(ys)>0 and len(xs)>0:
                obj_imgn_start_x = xs.min() 
                obj_imgn_start_y = ys.min()
                obj_imgn_end_x = xs.max()
                obj_imgn_end_y = ys.max()
            else:
                obj_imgn_start_x=0
                obj_imgn_start_y=0
                obj_imgn_end_x=0
                obj_imgn_end_y=0


            # mask region
            left_dist = zoom_c_x - obj_imgn_start_x
            right_dist = obj_imgn_end_x - zoom_c_x
            up_dist = zoom_c_y - obj_imgn_start_y
            down_dist = obj_imgn_end_y - zoom_c_y
            # crop_height = np.max([ratio * right_dist, ratio * left_dist, up_dist, down_dist]) * 2 * 1.4
            crop_height = np.max([ratio * right_dist, ratio * left_dist, up_dist, down_dist]) * 2 * (1+margin_ratio)
            crop_width = crop_height / ratio

            # affine transformation for PyTorch
            x1 = (zoom_c_x - crop_width / 2) * 2 / W - 1;
            x2 = (zoom_c_x + crop_width / 2) * 2 / W - 1;
            y1 = (zoom_c_y - crop_height / 2) * 2 / H - 1;
            y2 = (zoom_c_y + crop_height / 2) * 2 / H - 1;

            pts1 = np.float32([[x1, y1], [x1, y2], [x2, y1]])
            pts2 = np.float32([[-1, -1], [-1, 1], [1, -1]])
            affine_matrix = torch.tensor(cv2.getAffineTransform(pts2, pts1), device=mask.device, dtype=torch.float32)
            affine_matrices.append(affine_matrix)


            if with_intrinsic_transform:
                # affine transformation for PyTorch
                x1 = (zoom_c_x - crop_width / 2)
                x2 = (zoom_c_x + crop_width / 2)
                y1 = (zoom_c_y - crop_height / 2)
                y2 = (zoom_c_y + crop_height / 2)

                pts1 = np.float32([[x1, y1], [x1, y2], [x2, y1]])
                # pts2 = np.float32([[0, 0], [0, H-1], [W-1, 0]])
                pts2 = np.float32([[0, 0], [0, output_size[0]-1], [output_size[1]-1, 0]])
                # pts2 = np.float32([[0, 0], [0, 1], [1, 0]])
                intrinsic_matrix = torch.tensor(cv2.getAffineTransform(pts2, pts1), device=mask.device, dtype=torch.float32)
                intrinsic_matrices.append(intrinsic_matrix)
                
        if with_intrinsic_transform:
            return  torch.stack(affine_matrices, dim=0), torch.stack(intrinsic_matrices, dim=0)
        else:
            return torch.stack(affine_matrices, dim=0)

    def gen_zoom_crop_grids(self, fg_mask, K, T, output_size, model_center=[0,0,0], margin_ratio=0.4):
        ##Get the projected model center in image (assuming the model is zero-centered, which should be reconsidered!)  
        crop_center=K@T[:,:3,3:]
        crop_center = crop_center[:,:2]/crop_center[:,2:3]

        ##calculate affine transformation parameters
        affine_matrices, crop_intrinsic_transform=self.get_affine_transformation(fg_mask, crop_center=crop_center.detach().cpu().numpy(), with_intrinsic_transform=True, output_size=(output_size[-2], output_size[-1]),margin_ratio=margin_ratio ) 
        grids = F.affine_grid(affine_matrices, torch.Size(output_size) )
        ##Get cropped intrinsic_transform
        intrinsics_crop= torch.inverse(pose_padding(crop_intrinsic_transform) )@K

        return grids, intrinsics_crop

    # def forward(self, image, Ts, intrinsics, fea_3d=None, inds=None, Tj_gt=None, obj_cls=None, geofea_3d=None, geofea_2d=None):
    def forward(self, image, Ts, intrinsics, fea_3d=None, Tj_gt=None, obj_cls=None, geofea_3d=None, geofea_2d=None):
        #clear the history data
        self._clear()
        cfg = self.cfg

        RANDER_IMAGE_SIZE = get_cfg("BASIC").render_image_size
        ZOOM_CROP_SIZE=get_cfg("BASIC").zoom_crop_size 

        if cfg.RESCALE_IMAGES:
            images = 2 * (images / 255.0) - 1.0

        Tij_gt=[]
        syn_imgs=[]
        syn_depths=[]

        Ti = Ts
        Tij = Ti.copy().identity()

        for ren_iter in range(cfg.RENDER_ITER_COUNT):
            # update rendering params
            Ti = Tij*Ti # accumulate Ti 
            Tij.identity_() #set Tij to identity matrix at the begining of each ren_iter
            if self.legacy:
                Tij = Ti*Ti.inv() #set Tij to identity matrix at the begining of each ren_iter

            render_params = edict({
                    "K": intrinsics.detach(), 
                    "camera_extrinsics": Ti.matrix().detach().squeeze(1), 
                    "obj_cls": obj_cls,
                    "render_image_size": RANDER_IMAGE_SIZE, 
                })

            pc_depth = self.renderer.render_pointcloud(obj_cls, T=render_params.camera_extrinsics, K=render_params.K, 
                                    render_image_size=render_params.render_image_size)


            if self.cfg.ONLINE_CROP:
                #get the forground mask 
                fg_mask = pc_depth>0
                B,C,_,_= pc_depth.size()
                
                ############### Get zoom parameters ###############
                grids, intrinsics_crop = self.gen_zoom_crop_grids(fg_mask, render_params.K, render_params.camera_extrinsics, output_size=[B,C, *ZOOM_CROP_SIZE], model_center=None)

                ############### Render reference images ###############
                # Concatentate the 3D ctx feature "fea_3d" and 3d descriptor "geofea_3d" for feature rendering
                if geofea_3d is not None:
                    fea_3d_cat = torch.cat([fea_3d, geofea_3d ], dim=-1) # BxNxC
                else:
                    fea_3d_cat = fea_3d
                render_params.vert_attribute = fea_3d_cat #fea_3d
                render_params.K = intrinsics_crop.detach()
                render_params.render_image_size = ZOOM_CROP_SIZE
                ren_res = self.render(render_params, render_tex=True) 
                if geofea_3d is not None:
                    syn_img, cfea, geofea1 = torch.split(ren_res['syn_img'],[3, fea_3d.shape[-1], geofea_3d.shape[-1] ] ,dim=1)
                    syn_depth = ren_res['syn_depth']
                    
                else:
                    syn_img, cfea = torch.split(ren_res['syn_img'], [3, fea_3d.shape[-1] ] ,dim=1)
                    geofea1=None
                    syn_depth = ren_res['syn_depth']
                cfea = cfea*0.1 # balance the learning rate with the scale 0.1

                ## Crop and zoom images
                syn_image_crop = syn_img 
                image_crop= F.grid_sample(image, grids)
                cfea_crop= cfea 
                if geofea1 is not None and geofea_2d is not None:
                    # geofea1_crop = F.grid_sample(geofea1, grids)
                    geofea1_crop = geofea1
                    geofea2_crop = F.grid_sample(geofea_2d, grids)

                #Render again to get more accurate depth for supervisions in losses 
                #TODO: could be merged into the rendering process above. -> Done
                if self.legacy:#self.training:
                    depth_render_params=edict({
                            "K": intrinsics_crop.detach(), 
                            "camera_extrinsics": Ti.matrix().detach().squeeze(1), 
                            "obj_cls": obj_cls,
                            "render_image_size": ZOOM_CROP_SIZE, 
                        })
                    syn_depth=self.renderer.render_depth(obj_cls, T=depth_render_params.camera_extrinsics, K=depth_render_params.K, 
                                    render_image_size=depth_render_params.render_image_size, near=0.1, far=6)

            #for visualization only
            syn_imgs.append(syn_image_crop)
            syn_imgs.append(image_crop)

            # encode image features
            feats1, feats2=self.image_fea_enc(syn_image_crop, image_crop)
            # depths = torch.index_select(syn_depth, index=ii, dim=1) + EPS
            depths = syn_depth+EPS

            for i in range(cfg.ITER_COUNT):
                # save for loss calculation 
                self.intrinsics_history.append(intrinsics_crop)
                syn_depths.append(syn_depth)

                Tij = Tij.copy(stop_gradients=True)
                intrinsics_crop = intrinsics_crop.detach()
                
                #Get the projection in frame j of visible model points in frame i with the current relative pose estimation Tij
                reproj_coords, vmask = Tij.transform(
                    depths, intrinsics_crop, valid_mask=True)

                uniform_grids = coords_grid(depths)
                flow_init = torch.einsum( "...ijk->...kij", reproj_coords-uniform_grids[..., :2] ) * (depths>EPS) 
                flow = self.cf_net(feats1, feats2, flow_init=flow_init.squeeze(1), context_fea=cfea_crop, update_corr_fn=i==0)


                self.flow_history.append(flow)

                # Get the correspondences in frame j for each point in frame i, based on the current flow estimates
                if isinstance (flow, (list, tuple)): # flow net may return a list of flow maps
                    correspondence_target = torch.einsum("...ijk->...jki", flow[-1]) + uniform_grids[..., :2]
                else:
                    correspondence_target = torch.einsum("...ijk->...jki", flow) + uniform_grids[..., :2]
                
                # Optimize for the pose by minimizing errors between the constructed correspondence field 
                # (with the currently estimated pose) and the estimated correspondence field 
                if self.with_corr_weight and geofea1 is not None and geofea_2d is not None:
                    geofea2_crop_warpped =  F.grid_sample(geofea2_crop, normalize_coords_grid(correspondence_target).squeeze(1) )
                    corr_weight= torch.sum(geofea1_crop*geofea2_crop_warpped, dim=1,keepdim=True).permute(0,2,3,1)[:,None] #insert frame axis
                    corr_weight = torch.exp(-torch.abs(1-corr_weight)/self.sigma[0]) * (syn_depth>0)[...,None].float()
                else: 
                    corr_weight = weight
                
                Tij = Tij.reprojction_optim(
                    correspondence_target, corr_weight, depths, intrinsics_crop, num_iters=cfg.OPTIM_ITER_COUNT )

                reproj_coords, vmask1 = Tij.transform(
                    depths, intrinsics_crop, valid_mask=True)

                # For the loss calculation later
                self.residual_pose_history.append(Tij)

                self.Ti_history.append(Ti.copy(stop_gradients=True) )
                Tij_gt.append( (Tj_gt*Ti.inv()).copy(stop_gradients=True) )

                self.residual_history.append(
                    vmask*vmask1*(reproj_coords-correspondence_target))  # BxKxHxWx3

        # The final update of Ti 
        Ti = Tij*Ti
        return {
            "Tij": Tij,
            "Ti_pred": Ti,
            "intrinsics": intrinsics,
            "flow": self.flow_history[0],
            "vmask": syn_depth>0, 
            "weight": torch.einsum("...ijk->...kij", corr_weight), 
            "syn_depth": syn_depths, #ren_res['syn_depth'],
            "syn_img": syn_imgs+[image_crop, cfea_crop[:,:3]*10,geofea1[:,:3], geofea2_crop[:,:3]],
            "Tij_gt" : Tij_gt
        }

    def compute_loss(self, Tij_gts, depths, intrinsics, loss='l1', log_error=True, loss3d=None, ):

        total_loss = 0.0
        for i in range(len(self.residual_pose_history)):
            intrinsics = self.intrinsics_history[i]

            depth, intrinsics = rescale_depths_and_intrinsics(depths[i], intrinsics, downscale=1)

            Tij = self.residual_pose_history[i]

            Gij = Tij_gts[i] 

            # intrinsics_pred = intrinsics
            zstar = depth + EPS
            flow_pred, valid_mask_pred = Tij.induced_flow(
                zstar, intrinsics, valid_mask=True)  
            flow_star, valid_mask_star = Gij.induced_flow(
                zstar, intrinsics, valid_mask=True)

            valid_mask = valid_mask_pred * valid_mask_star

            #3D alignment loss 
            loss_3d_proj = 0 
            if loss3d is not None:
                Tj_pred=Tij*self.Ti_history[i]
                Tj_gt=Gij*self.Ti_history[i]
                loss_3d_proj = loss3d(
                    R_pred=Tj_pred.G[:, 0, :3, :3], t_pred=Tj_pred.G[:, 0, :3, 3], R_tgt=Tj_gt.G[:, 0, :3, :3], t_tgt=Tj_gt.G[:, 0, :3, 3])

            # flow loss
            if isinstance( self.flow_history[0], (list, tuple)):
                #squeeze the frame dimmension
                self.flow_history[i] = [self.flow_history[i][f].squeeze(1) for f in range(len(self.flow_history[i])) ]
                flow_mask = valid_mask.squeeze(1).squeeze(-1)
                loss_flow,_ = raft_sequence_flow_loss(self.flow_history[i], flow_gt=torch.einsum("...ijk->...kij", flow_star.squeeze(1)), valid= flow_mask, gamma=0.8, max_flow=MAX_FLOW)
            else:
                raise NotImplementedError
            
            # reprojection loss 
            reproj_diff = valid_mask * \
                torch.clamp(
                    torch.abs(flow_pred - flow_star), -MAX_ERROR, MAX_ERROR)
            reproj_loss = torch.mean(reproj_diff)


            total_loss +=self.cfg.get("TRAIN_PCALIGN_WEIGHT", 1)*loss_3d_proj+ self.cfg.TRAIN_FLOW_WEIGHT* loss_flow + self.cfg.TRAIN_REPROJ_WEIGHT*reproj_loss
        
        # clear the intermediate values
        self._clear()

        return {
            "total_loss": total_loss,
            "reproj_loss": reproj_loss,
            "flow_loss":loss_flow,
            "loss_3d_proj": loss_3d_proj,
            "valid_mask": valid_mask,
            "Tij": Tj_pred.G,
            "Gij": Tj_gt.G
        }

import os
import numpy as np
from plyfile import PlyData
# from utils import icp_utils
from data.linemod import linemod_config
from thirdparty.vsd import inout
from thirdparty.nn import nn_utils
from utils.img_utils import read_depth
from thirdparty.kpconv.lib.utils import square_distance
from utils.geometric import rotation_angle
from utils.visualize import *
# from thirdparty.fps import fps_utils
import torch
import open3d as o3d
from transforms3d.quaternions import mat2quat, quat2mat, qmult
# import data.bop_ycb.ycb_config as ycb_config #import bop_ycb_class2idx, model_info

def get_ply_model(model_path, scale=1):
    ply = PlyData.read(model_path)
    data = ply.elements[0].data
    x = data['x']*scale
    y = data['y']*scale
    z = data['z']*scale
    model = np.stack([x, y, z], axis=-1)
    return model


def project(xyz, K, RT):
    """
    xyz: [N, 3]
    K: [3, 3]
    RT: [3, 4]
    """
    xyz = np.dot(xyz, RT[:, :3].T) + RT[:, 3:].T
    xyz = np.dot(xyz, K.T)
    xy = xyz[:, :2] / xyz[:, 2:]
    return xy


def find_nearest_point_idx(ref_pts, que_pts):
    assert(ref_pts.shape[1] == que_pts.shape[1] and 1 < que_pts.shape[1] <= 3)
    pn1 = ref_pts.shape[0]
    pn2 = que_pts.shape[0]
    dim = ref_pts.shape[1]

    ref_pts = np.ascontiguousarray(ref_pts[None, :, :], np.float32)
    que_pts = np.ascontiguousarray(que_pts[None, :, :], np.float32)
    idxs = np.zeros([1, pn2], np.int32)

    ref_pts_ptr = ffi.cast('float *', ref_pts.ctypes.data)
    que_pts_ptr = ffi.cast('float *', que_pts.ctypes.data)
    idxs_ptr = ffi.cast('int *', idxs.ctypes.data)
    lib.findNearestPointIdxLauncher(
        ref_pts_ptr, que_pts_ptr, idxs_ptr, 1, pn1, pn2, dim, 0)

    return idxs[0]


class LineMODEvaluator:
    def __init__(self, class_name, result_dir, icp_refine=False):

        # self.result_dir = os.path.join(result_dir, cfg.test.dataset)
        self.result_dir = os.path.join(result_dir, "LINEMOD")
        os.system('mkdir -p {}'.format(self.result_dir))


        # data_root = args['data_root']
        # cls = cfg.cls_type
        self.class_name = class_name
        self.icp_refine = icp_refine

        # model_path = os.path.join(os.path.dirname(os.path.abspath(
        #     __file__)), '../EXPDATA/LINEMOD', class_name, class_name + '.ply')
        model_path = os.path.join(os.path.dirname(os.path.abspath(
            __file__)), '../EXPDATA/LM6d_converted/models', class_name, class_name + '.ply')
        # self.model = pvnet_data_utils.get_ply_model(model_path)
        self.model = get_ply_model(model_path)
        self.diameter = linemod_config.diameters[class_name] / 100

        self.proj2d = []
        self.add = []
        self.adds = [] #force sym
        self.add2 = []
        self.add5 = []
        self.cmd5 = []

        self.icp_proj2d = []
        self.icp_add = []
        self.icp_cmd5 = []

        self.mask_ap = []
        self.pose_preds=[]

        self.height = 480
        self.width = 640

        model = inout.load_ply(model_path)
        model['pts'] = model['pts'] * 1000
        self.icp_refiner = icp_utils.ICPRefiner(
            model, (self.width, self.height)) if icp_refine else None

    def projection_2d(self, pose_pred, pose_targets, K, icp=False, threshold=5):
        model_2d_pred = project(self.model, K, pose_pred)
        model_2d_targets = project(self.model, K, pose_targets)
        proj_mean_diff = np.mean(np.linalg.norm(
            model_2d_pred - model_2d_targets, axis=-1))
        if icp:
            self.icp_proj2d.append(proj_mean_diff < threshold)
        else:
            self.proj2d.append(proj_mean_diff < threshold)

    def projection_2d_sym(self, pose_pred, pose_targets, K, threshold=5):
        model_2d_pred = project(self.model, K, pose_pred)
        model_2d_targets = project(self.model, K, pose_targets)
        proj_mean_diff=np.mean(find_nearest_point_distance(model_2d_pred,model_2d_targets))

        self.proj_mean_diffs.append(proj_mean_diff)
        self.projection_2d_recorder.append(proj_mean_diff < threshold)

    def add2_metric(self, pose_pred, pose_targets, icp=False, syn=False, percentage=0.02):
        diameter = self.diameter * percentage
        model_pred = np.dot(self.model, pose_pred[:, :3].T) + pose_pred[:, 3]
        model_targets = np.dot(
            self.model, pose_targets[:, :3].T) + pose_targets[:, 3]

        if syn:
            idxs = nn_utils.find_nearest_point_idx(model_pred, model_targets)
            # idxs = find_nearest_point_idx(model_pred, model_targets)
            mean_dist = np.mean(np.linalg.norm(
                model_pred[idxs] - model_targets, 2, 1))
        else:
            mean_dist = np.mean(np.linalg.norm(
                model_pred - model_targets, axis=-1))

        if icp:
            self.icp_add.append(mean_dist < diameter)
        else:
            self.add2.append(mean_dist < diameter)

    def add5_metric(self, pose_pred, pose_targets, icp=False, syn=False, percentage=0.05):
        diameter = self.diameter * percentage
        model_pred = np.dot(self.model, pose_pred[:, :3].T) + pose_pred[:, 3]
        model_targets = np.dot(
            self.model, pose_targets[:, :3].T) + pose_targets[:, 3]

        if syn:
            idxs = nn_utils.find_nearest_point_idx(model_pred, model_targets)
            # idxs = find_nearest_point_idx(model_pred, model_targets)
            mean_dist = np.mean(np.linalg.norm(
                model_pred[idxs] - model_targets, 2, 1))
        else:
            mean_dist = np.mean(np.linalg.norm(
                model_pred - model_targets, axis=-1))

        if icp:
            self.icp_add.append(mean_dist < diameter)
        else:
            self.add5.append(mean_dist < diameter)

    
    def add_metric(self, pose_pred, pose_targets, icp=False, syn=False, percentage=0.1):
        diameter = self.diameter * percentage
        model_pred = np.dot(self.model, pose_pred[:, :3].T) + pose_pred[:, 3]
        model_targets = np.dot(
            self.model, pose_targets[:, :3].T) + pose_targets[:, 3]

        if syn:
            idxs = nn_utils.find_nearest_point_idx(model_pred, model_targets)
            # idxs = find_nearest_point_idx(model_pred, model_targets)
            mean_dist = np.mean(np.linalg.norm(
                model_pred[idxs] - model_targets, 2, 1))
        else:
            mean_dist = np.mean(np.linalg.norm(
                model_pred - model_targets, axis=-1))

        if icp:
            self.icp_add.append(mean_dist < diameter)
        else:
            self.add.append(mean_dist < diameter)

    def cm_degree_5_metric(self, pose_pred, pose_targets, icp=False):
        translation_distance = np.linalg.norm(
            pose_pred[:, 3] - pose_targets[:, 3]) * 100
        rotation_diff = np.dot(pose_pred[:, :3], pose_targets[:, :3].T)
        trace = np.trace(rotation_diff)
        trace = trace if trace <= 3 else 3
        angular_distance = np.rad2deg(np.arccos((trace - 1.) / 2.))
        if icp:
            self.icp_cmd5.append(translation_distance <
                                 5 and angular_distance < 5)
        else:
            self.cmd5.append(translation_distance < 5 and angular_distance < 5)

    def mask_iou(self, output, batch):
        mask_pred = torch.argmax(output['seg'], dim=1)[
            0].detach().cpu().numpy()
        mask_gt = batch['mask'][0].detach().cpu().numpy()
        iou = (mask_pred & mask_gt).sum() / (mask_pred | mask_gt).sum()
        self.mask_ap.append(iou > 0.7)

    def icp_refine(self, pose_pred, anno, output, K):
        depth = read_depth(anno['depth_path'])
        mask = torch.argmax(output['seg'], dim=1)[0].detach().cpu().numpy()
        if pose_pred[2, 3] <= 0:
            return pose_pred
        depth[mask != 1] = 0
        pose_pred_tmp = pose_pred.copy()
        pose_pred_tmp[:3, 3] = pose_pred_tmp[:3, 3] * 1000

        R_refined, t_refined = self.icp_refiner.refine(
            depth, pose_pred_tmp[:3, :3], pose_pred_tmp[:3, 3], K.copy(), depth_only=True, max_mean_dist_factor=5.0)
        R_refined, _ = self.icp_refiner.refine(
            depth, R_refined, t_refined, K.copy(), no_depth=True)

        pose_pred = np.hstack((R_refined, t_refined.reshape((3, 1)) / 1000))

        return pose_pred


    def icp_refine_(self, pose, anno, output):
        depth = read_depth(anno['depth_path']).astype(np.uint16)
        mask = torch.argmax(output['seg'], dim=1)[0].detach().cpu().numpy()
        mask = mask.astype(np.int32)
        pose = pose.astype(np.float32)

        poses = np.zeros([1, 7], dtype=np.float32)
        poses[0, :4] = mat2quat(pose[:, :3])
        poses[0, 4:] = pose[:, 3]

        poses_new = np.zeros([1, 7], dtype=np.float32)
        poses_icp = np.zeros([1, 7], dtype=np.float32)

        fx = 572.41140
        fy = 573.57043
        px = 325.26110
        py = 242.04899
        zfar = 6.0
        znear = 0.25
        factor = 1000.0
        error_threshold = 0.01

        rois = np.zeros([1, 6], dtype=np.float32)
        rois[:, :] = 1

        self.icp_refiner.solveICP(mask, depth,
                                  self.height, self.width,
                                  fx, fy, px, py,
                                  znear, zfar,
                                  factor,
                                  rois.shape[0], rois,
                                  poses, poses_new, poses_icp,
                                  error_threshold
                                  )

        pose_icp = np.zeros([3, 4], dtype=np.float32)
        pose_icp[:, :3] = quat2mat(poses_icp[0, :4])
        pose_icp[:, 3] = poses_icp[0, 4:]

        return pose_icp

    def summarize(self):
        proj2d = np.mean(self.proj2d)
        add = np.mean(self.add)
        # adds = np.mean(self.adds)
        add2 = np.mean(self.add2)
        add5 = np.mean(self.add5)
        cmd5 = np.mean(self.cmd5)
        ap = np.mean(self.mask_ap)
        seq_len=len(self.add)
        print('2d projections metric: {}'.format(proj2d))
        print('ADD metric: {}'.format(add))
        print('ADD2 metric: {}'.format(add2))
        print('ADD5 metric: {}'.format(add5))
        # print('ADDS metric: {}'.format(adds))
        print('5 cm 5 degree metric: {}'.format(cmd5))
        print('mask ap70: {}'.format(ap))
        print('seq_len: {}'.format(seq_len))
        # if cfg.test.icp:
        if self.icp_refine:
            print('2d projections metric after icp: {}'.format(
                np.mean(self.icp_proj2d)))
            print('ADD metric after icp: {}'.format(np.mean(self.icp_add)))
            print('5 cm 5 degree metric after icp: {}'.format(
                np.mean(self.icp_cmd5)))
        self.proj2d = []
        self.add = []
        self.add2 = []
        self.add5 = []
        # self.adds = []
        self.cmd5 = []
        self.mask_ap = []
        self.icp_proj2d = []    
        self.icp_add = []
        self.icp_cmd5 = []
        

        #save pose predictions
        if len(self.pose_preds)> 0:
            np.save(f"{self.class_name}_pose_preds.npy",self.pose_preds)
        self.pose_preds=[]

        return {'proj2d': proj2d, 'add': add, 'add2': add2, 'add5': add5,'cmd5': cmd5, 'ap': ap, "seq_len": seq_len}



    def evaluate_rnnpose(self, preds_dict, example): # sample_correspondence_pairs=False, direct_align=False, use_cnnpose=True):
        len_src_f = example['stack_lengths'][0][0]
        # lifted_points = example['lifted_points'].squeeze(0)
        assert len( example['lifted_points']) == 1, "TODO: support bs>1"
        lifted_points = example['lifted_points'][0].squeeze(0)
        model_points = example['original_model_points'][:len_src_f]

        K = example["K"].cpu().numpy().squeeze()
        R_pred = preds_dict['Ti_pred'].G[:,0, :3,:3].squeeze().detach().cpu().numpy()
        t_pred = preds_dict['Ti_pred'].G[:,0, :3,3:].squeeze(0).detach().cpu().numpy()
        pose_pred= preds_dict['Ti_pred'].G[:,0, :3].squeeze().detach().cpu().numpy()
#             print(example['POSECNN_RT'].dtype, example['rendered_RT'].dtype, flush=True)
#             R_pred = example['POSECNN_RT'][:,:3,:3].squeeze().detach().cpu().numpy()
#             t_pred = example['POSECNN_RT'][:,:3,3:].squeeze(0).detach().cpu().numpy()
#             pose_pred= example['POSECNN_RT'][:, :3].squeeze().detach().cpu().numpy()


        pose_gt = example['original_RT'].squeeze()[:3].cpu().numpy()
        
        
        ang_err = rotation_angle(pose_gt[:3, :3], R_pred)
        trans_err = np.linalg.norm(t_pred-pose_gt[:3, -1:])  # 3x1

        if self.class_name in ['eggbox', 'glue']:
            self.add_metric(pose_pred, pose_gt, syn=True)
            self.add2_metric(pose_pred, pose_gt, syn=True)
            self.add5_metric(pose_pred, pose_gt, syn=True)
        else:
            self.add_metric(pose_pred, pose_gt)
            self.add2_metric(pose_pred, pose_gt)
            self.add5_metric(pose_pred, pose_gt)

        self.projection_2d(pose_pred, pose_gt, K=linemod_config.linemod_K)
        self.cm_degree_5_metric(pose_pred, pose_gt)
        # self.mask_iou(output, batch)

        # vis
        pc_proj_vis = vis_pointclouds_cv2((pose_gt[:3, :3]@model_points.cpu().numpy(
        ).T+pose_gt[:3, -1:]).T, example["K"].cpu().numpy().squeeze(), [480,640])
        pc_proj_vis_pred = vis_pointclouds_cv2((pose_pred[:3, :3]@model_points.cpu().numpy(
        ).T+pose_pred[:3, -1:]).T, example["K"].cpu().numpy().squeeze(), [ 480, 640])


        return {
            "ang_err": ang_err,
            "trans_err": trans_err,
            "pnp_inliers": -1,#len(inliers),
            "pc_proj_vis": pc_proj_vis,
            "pc_proj_vis_pred": pc_proj_vis_pred,
            "keypoints_2d_vis": np.zeros_like(pc_proj_vis_pred) #keypoints_2d_vis
        }

        


# class YCBEvaluator:
#     def __init__(self, class_name, result_dir, icp_refine=False):

#         self.result_dir = os.path.join(result_dir, "LINEMOD")
#         os.system('mkdir -p {}'.format(self.result_dir))

#         self.class_name = class_name
#         self.icp_refine = icp_refine
        
#         model_path = os.path.join(os.path.dirname(os.path.abspath(
#             __file__)), '../EXPDATA/BOP_YCB/models', f'obj_{ycb_config.bop_ycb_class2idx[class_name]:06d}.ply' )
#         self.model = get_ply_model(model_path, scale=0.001)
#         # self.diameter = linemod_config.diameters[class_name] / 100
#         self.diameter = ycb_config.model_info[ str(ycb_config.bop_ycb_class2idx[class_name]) ]["diameter"]*0.001  # in mm # / 1000

#         self.proj2d = []
#         self.add = []
#         self.adds=[]
#         self.cmd5 = []
#         self.add_dist=[]
#         self.adds_dist=[]

#         self.icp_proj2d = []
#         self.icp_add = []
#         self.icp_cmd5 = []

#         self.mask_ap = []
#         self.pose_preds=[]

#         self.height = 480
#         self.width = 640

#         model = inout.load_ply(model_path)
#         model['pts'] = model['pts'] * 1000
#         # self.icp_refiner = icp_utils.ICPRefiner(model, (self.width, self.height)) if cfg.test.icp else None
#         self.icp_refiner = icp_utils.ICPRefiner(
#             model, (self.width, self.height)) if icp_refine else None
#         self.direct_align_module = DirectAlignment(None)
#         # if cfg.test.icp:
#         #     self.icp_refiner = ext_.Synthesizer(os.path.realpath(model_path))
#         #     self.icp_refiner.setup(self.width, self.height)

#     def projection_2d(self, pose_pred, pose_targets, K, icp=False, threshold=5):
#         model_2d_pred = project(self.model, K, pose_pred)
#         model_2d_targets = project(self.model, K, pose_targets)
#         proj_mean_diff = np.mean(np.linalg.norm(
#             model_2d_pred - model_2d_targets, axis=-1))
#         if icp:
#             self.icp_proj2d.append(proj_mean_diff < threshold)
#         else:
#             self.proj2d.append(proj_mean_diff < threshold)

#     def projection_2d_sym(self, pose_pred, pose_targets, K, threshold=5):
#         model_2d_pred = project(self.model, K, pose_pred)
#         model_2d_targets = project(self.model, K, pose_targets)
#         proj_mean_diff=np.mean(find_nearest_point_distance(model_2d_pred,model_2d_targets))

#         self.proj_mean_diffs.append(proj_mean_diff)
#         self.projection_2d_recorder.append(proj_mean_diff < threshold)

#     def add_metric(self, pose_pred, pose_targets, icp=False, syn=False, percentage=0.1):
#         diameter = self.diameter * percentage
#         model_pred = np.dot(self.model, pose_pred[:, :3].T) + pose_pred[:, 3]
#         model_targets = np.dot(
#             self.model, pose_targets[:, :3].T) + pose_targets[:, 3]

#         if syn:
#             idxs = nn_utils.find_nearest_point_idx(model_pred, model_targets)
#             # idxs = find_nearest_point_idx(model_pred, model_targets)
#             mean_dist = np.mean(np.linalg.norm(
#                 model_pred[idxs] - model_targets, 2, 1))
#         else:
#             mean_dist = np.mean(np.linalg.norm(
#                 model_pred - model_targets, axis=-1))
#         self.add_dist.append(mean_dist)
#         if icp:
#             self.icp_add.append(mean_dist < diameter)
#         else:
#             self.add.append(mean_dist < diameter)
#     def auc_add(self, max_thresh=0.1):
#         add_dist = np.array(self.add_dist)
#         interval=0.001
#         acc=0
#         for k in range(int(max_thresh/interval)):
#             acc+= interval* np.sum( ((k+1)*interval)>=add_dist)/ add_dist.shape[0]

#         return acc/max_thresh
#     def auc_adds(self, max_thresh=0.1):
#         add_dist = np.array(self.adds_dist)
#         interval=0.001
#         acc=0
#         for k in range(int(max_thresh/interval)):
#             acc+= interval* np.sum( ((k+1)*interval)>=add_dist )/ add_dist.shape[0]

#         return acc/max_thresh
#     def adds_metric(self, pose_pred, pose_targets, icp=False, syn=False, percentage=0.1):
#         diameter = self.diameter * percentage
#         model_pred = np.dot(self.model, pose_pred[:, :3].T) + pose_pred[:, 3]
#         model_targets = np.dot(
#             self.model, pose_targets[:, :3].T) + pose_targets[:, 3]

#         if syn:
#             idxs = nn_utils.find_nearest_point_idx(model_pred, model_targets)
#             # idxs = find_nearest_point_idx(model_pred, model_targets)
#             mean_dist = np.mean(np.linalg.norm(
#                 model_pred[idxs] - model_targets, 2, 1))
#         else:
#             mean_dist = np.mean(np.linalg.norm(
#                 model_pred - model_targets, axis=-1))
#         self.adds_dist.append(mean_dist)
#         if icp:
#             self.icp_add.append(mean_dist < diameter)
#         else:
#             self.adds.append(mean_dist < diameter)

#     def cm_degree_5_metric(self, pose_pred, pose_targets, icp=False):
#         translation_distance = np.linalg.norm(
#             pose_pred[:, 3] - pose_targets[:, 3]) * 100
#         rotation_diff = np.dot(pose_pred[:, :3], pose_targets[:, :3].T)
#         trace = np.trace(rotation_diff)
#         trace = trace if trace <= 3 else 3
#         angular_distance = np.rad2deg(np.arccos((trace - 1.) / 2.))
#         if icp:
#             self.icp_cmd5.append(translation_distance <
#                                  5 and angular_distance < 5)
#         else:
#             self.cmd5.append(translation_distance < 5 and angular_distance < 5)

#     def mask_iou(self, output, batch):
#         mask_pred = torch.argmax(output['seg'], dim=1)[
#             0].detach().cpu().numpy()
#         mask_gt = batch['mask'][0].detach().cpu().numpy()
#         iou = (mask_pred & mask_gt).sum() / (mask_pred | mask_gt).sum()
#         self.mask_ap.append(iou > 0.7)

#     def icp_refine(self, pose_pred, anno, output, K):
#         depth = read_depth(anno['depth_path'])
#         mask = torch.argmax(output['seg'], dim=1)[0].detach().cpu().numpy()
#         if pose_pred[2, 3] <= 0:
#             return pose_pred
#         depth[mask != 1] = 0
#         pose_pred_tmp = pose_pred.copy()
#         pose_pred_tmp[:3, 3] = pose_pred_tmp[:3, 3] * 1000

#         R_refined, t_refined = self.icp_refiner.refine(
#             depth, pose_pred_tmp[:3, :3], pose_pred_tmp[:3, 3], K.copy(), depth_only=True, max_mean_dist_factor=5.0)
#         R_refined, _ = self.icp_refiner.refine(
#             depth, R_refined, t_refined, K.copy(), no_depth=True)

#         pose_pred = np.hstack((R_refined, t_refined.reshape((3, 1)) / 1000))

#         return pose_pred


#     def icp_refine_(self, pose, anno, output):
#         depth = read_depth(anno['depth_path']).astype(np.uint16)
#         mask = torch.argmax(output['seg'], dim=1)[0].detach().cpu().numpy()
#         mask = mask.astype(np.int32)
#         pose = pose.astype(np.float32)

#         poses = np.zeros([1, 7], dtype=np.float32)
#         poses[0, :4] = mat2quat(pose[:, :3])
#         poses[0, 4:] = pose[:, 3]

#         poses_new = np.zeros([1, 7], dtype=np.float32)
#         poses_icp = np.zeros([1, 7], dtype=np.float32)

#         fx = 572.41140
#         fy = 573.57043
#         px = 325.26110
#         py = 242.04899
#         zfar = 6.0
#         znear = 0.25
#         factor = 1000.0
#         error_threshold = 0.01

#         rois = np.zeros([1, 6], dtype=np.float32)
#         rois[:, :] = 1

#         self.icp_refiner.solveICP(mask, depth,
#                                   self.height, self.width,
#                                   fx, fy, px, py,
#                                   znear, zfar,
#                                   factor,
#                                   rois.shape[0], rois,
#                                   poses, poses_new, poses_icp,
#                                   error_threshold
#                                   )

#         pose_icp = np.zeros([3, 4], dtype=np.float32)
#         pose_icp[:, :3] = quat2mat(poses_icp[0, :4])
#         pose_icp[:, 3] = poses_icp[0, 4:]

#         return pose_icp

    

#     def summarize(self):
#         proj2d = np.mean(self.proj2d)
#         add = np.mean(self.add)
#         adds = np.mean(self.adds)
#         cmd5 = np.mean(self.cmd5)
#         ap = np.mean(self.mask_ap)
#         try:
#             auc_add=self.auc_add(max_thresh=0.1)
#         except:
#             auc_add=0
#         try:
#             auc_adds=self.auc_adds(max_thresh=0.1)
#         except:
#             auc_adds=0
#         seq_len=len(self.add)
#         print('2d projections metric: {}'.format(proj2d))
#         print('ADD metric: {}'.format(add))
#         print('AUC ADD metric: {}'.format(auc_add))
#         print('ADDS metric: {}'.format(adds))
#         print('AUC ADDS metric: {}'.format(auc_adds))
#         print('5 cm 5 degree metric: {}'.format(cmd5))
#         print('mask ap70: {}'.format(ap))
#         print('seq_len: {}'.format(seq_len))
#         # if cfg.test.icp:
#         if self.icp_refine:
#             print('2d projections metric after icp: {}'.format(
#                 np.mean(self.icp_proj2d)))
#             print('ADD metric after icp: {}'.format(np.mean(self.icp_add)))
#             print('5 cm 5 degree metric after icp: {}'.format(
#                 np.mean(self.icp_cmd5)))
#         self.proj2d = []
#         self.add = []
#         self.adds = []
#         self.cmd5 = []
#         self.mask_ap = []
#         self.icp_proj2d = []    
#         self.icp_add = []
#         self.icp_cmd5 = []
#         self.add_dist = []
#         self.adds_dist = []
        

#         #save pose predictions
#         if len(self.pose_preds)> 0:
#             np.save(f"{self.class_name}_pose_preds.npy",self.pose_preds)
#         self.pose_preds=[]

#         return {'proj2d': proj2d, 'add': add, 'adds': adds,'cmd5': cmd5, 'ap': ap, "seq_len": seq_len}

#     def evaluate_flowpose(self, preds_dict, example, sample_correspondence_pairs=False, direct_align=False, use_cnnpose=True):
#         len_src_f = example['stack_lengths'][0][0]
#         # lifted_points = example['lifted_points'].squeeze(0)
#         assert len( example['lifted_points']) == 1, "TODO: support bs>1"
#         lifted_points = example['lifted_points'][0].squeeze(0)
#         model_points = example['original_model_points'][:len_src_f]
#         K = example["K"].cpu().numpy().squeeze()

#         if not use_cnnpose: # use pnp 
#             len_src_f = example['stack_lengths'][0][0]
#             descriptors_2d = preds_dict['descriptors_2d']
#             descriptors_3d = preds_dict['descriptors_3d'][:len_src_f]

#             mask = cv2.erode(( example['depth'].detach().cpu().numpy().squeeze()>0).astype(np.uint8)*255, kernel=np.ones([3,3], np.uint8),iterations = 1)
#             mask=torch.tensor(mask, device=descriptors_3d.device)
#             ys_, xs_ = torch.nonzero(mask, as_tuple=True)

#             # ys_, xs_ = torch.nonzero(example['depth'].squeeze(), as_tuple=True)
#             descriptors_2d = descriptors_2d[:, :,
#                                             ys_, xs_].squeeze().permute([1, 0])
#             img_coods = torch.stack([xs_, ys_], dim=-1)



#             if sample_correspondence_pairs:
#                 correspondences_2d3d = example['correspondences_2d3d'].squeeze()
#                 if(correspondences_2d3d.size(0) > 256):
#                     choice = np.random.permutation(
#                         correspondences_2d3d.size(0))[:256]
#                     correspondences_2d3d = correspondences_2d3d[choice]

#                 src_idx = correspondences_2d3d[:, 0]
#                 tgt_idx = correspondences_2d3d[:, 1]
#             else:
#                 # src_idx = np.random.permutation(len(xs_))[:256]
#                 # src_idx = np.random.permutation(len(xs_))[:256]
#                 _, idx = fps_utils.farthest_point_sampling_withidx(np.stack([xs_.cpu().numpy(
#                 ), ys_.cpu().numpy(), np.zeros_like(xs_.cpu().numpy())], axis=-1), 256, False)
#                 # _, idx = fps_utils.farthest_point_sampling_withidx(np.stack([xs_.cpu().numpy(
#                 # ), ys_.cpu().numpy(), np.zeros_like(xs_.cpu().numpy())], axis=-1), 1024, False)
#                 # src_idx = np.arange(len(xs_))[::len(xs_)//256]
#                 src_idx = np.arange(len(xs_))[idx]
#                 tgt_idx = np.arange(len(model_points))

#             src_pcd, tgt_pcd = lifted_points[src_idx], model_points[tgt_idx]
#             img_coods = img_coods[src_idx]
#             src_feats, tgt_feats = descriptors_2d[src_idx], descriptors_3d[tgt_idx]

#             feats_dist = torch.sqrt(square_distance(
#                 src_feats[None, :, :], tgt_feats[None, :, :], normalised=True)).squeeze(0)

#             _, sel_idx = torch.min(feats_dist, -1)
#             K = example["K"].cpu().numpy().squeeze()
#             try:
#                 # retval, R_pred,t_pred, inliers =cv2.solvePnPRansac(tgt_pcd[sel_idx].cpu().numpy(), img_coods.cpu().numpy().astype(np.float32), K,distCoeffs=np.zeros(4),reprojectionError=1)
#                 retval, R_pred, t_pred, inliers = cv2.solvePnPRansac(tgt_pcd[sel_idx].cpu().numpy(), img_coods.cpu(
#                 ).numpy().astype(np.float32), K, distCoeffs=np.zeros(4), reprojectionError=1, iterationsCount=1000)

#                 if inliers is None:
#                     raise ValueError
#             except:
#                 # try:
#                 print("PNP RANSAC reprojectionError threshold =3")
#                 retval, R_pred, t_pred, inliers = cv2.solvePnPRansac(tgt_pcd[sel_idx].cpu().numpy(), img_coods.cpu(
#                 ).numpy().astype(np.float32), K, distCoeffs=np.zeros(4), reprojectionError=3, iterationsCount=1000)
    
#             R_pred, _ = cv2.Rodrigues(R_pred)
#             pose_pred = np.concatenate([R_pred, t_pred], axis=-1)
#         else:
#             K = example["K"].cpu().numpy().squeeze()
#             R_pred = preds_dict['Ti_pred'].G[:,0, :3,:3].squeeze().detach().cpu().numpy()
#             t_pred = preds_dict['Ti_pred'].G[:,0, :3,3:].squeeze(0).detach().cpu().numpy()
#             pose_pred= preds_dict['Ti_pred'].G[:,0, :3].squeeze().detach().cpu().numpy()
# #             print(example['POSECNN_RT'].dtype, example['rendered_RT'].dtype, flush=True)
# #             R_pred = example['POSECNN_RT'][:,:3,:3].squeeze().detach().cpu().numpy()
# #             t_pred = example['POSECNN_RT'][:,:3,3:].squeeze(0).detach().cpu().numpy()
# #             pose_pred= example['POSECNN_RT'][:, :3].squeeze().detach().cpu().numpy()


#         # pose_gt = example['RT'].squeeze()[:3].cpu().numpy()
#         pose_gt = example['original_RT'].squeeze()[:3].cpu().numpy()
        
        
#         ang_err = rotation_angle(pose_gt[:3, :3], R_pred)
#         trans_err = np.linalg.norm(t_pred-pose_gt[:3, -1:])  # 3x1

#         if self.class_name in ['024_bowl', '036_wood_block', '051_large_clamp', '052_extra_large_clamp', '061_foam_brick']:
#             self.add_metric(pose_pred, pose_gt, syn=True)
#         else:
#             self.add_metric(pose_pred, pose_gt)
#         self.adds_metric(pose_pred, pose_gt, syn=True)
#         self.projection_2d(pose_pred, pose_gt, K=linemod_config.linemod_K)
#         self.cm_degree_5_metric(pose_pred, pose_gt)

#         # self.mask_iou(output, batch)

#         # vis
#         pc_proj_vis = vis_pointclouds_cv2((pose_gt[:3, :3]@model_points.cpu().numpy(
#         ).T+pose_gt[:3, -1:]).T, example["K"].cpu().numpy().squeeze(), [480,640])
#         pc_proj_vis_pred = vis_pointclouds_cv2((pose_pred[:3, :3]@model_points.cpu().numpy(
#         ).T+pose_pred[:3, -1:]).T, example["K"].cpu().numpy().squeeze(), [ 480, 640])


#         if trans_err > 0.5:
#             print("translation err>0.5")
#             # cv2.imwrite(f'tmp/{len(self.add)}.png',
#             #             keypoints_2d_vis[..., ::-1]*255,)
#             # torch.save( example, f'tmp/{len(self.add)}.pt' )

#         return {
#             "ang_err": ang_err,
#             "trans_err": trans_err,
#             "pnp_inliers": -1,#len(inliers),
#             "pc_proj_vis": pc_proj_vis,
#             "pc_proj_vis_pred": pc_proj_vis_pred,
#             "keypoints_2d_vis": np.zeros_like(pc_proj_vis_pred) #keypoints_2d_vis
#         }

import open3d as o3d
import copy
import os

import pathlib
import pickle
import time
from collections import defaultdict
from functools import partial

import cv2
import numpy as np
import quaternion
from skimage import io as imgio
from utils.timer import simple_timer

import matplotlib.pyplot as plt
from collections.abc import Iterable
import torch
import torch.nn.functional as F
import quaternion

from functools import partial
import thirdparty.kpconv.cpp_wrappers.cpp_subsampling.grid_subsampling as cpp_subsampling
import thirdparty.kpconv.cpp_wrappers.cpp_neighbors.radius_neighbors as cpp_neighbors
from thirdparty.kpconv.lib.timer import Timer
from utils.geometric import range_to_depth, mask_depth_to_point_cloud
from utils.furthest_point_sample import fragmentation_fps
from utils.rand_utils import truncated_normal



def merge_batch(batch_list):
    # [batch][key][seq]->example[key][seq][batch]
    # Or [batch][key]->example[key][batch]
    example_merged = defaultdict(list)
    for example in batch_list:  # batch dim
        for k, v in example.items():  # key dim
            # assert isinstance(v, list)
            if isinstance(v, list):
                seq_len = len(v)
                if k not in example_merged:
                    example_merged[k] = [[] for i in range(seq_len)]
                for i, vi in enumerate(v):  # seq dim
                    example_merged[k][i].append(vi)

            else:
                example_merged[k].append(v)

    ret = {}
    for key, elems in example_merged.items():
        if key in ['model_points', "original_model_points", 'visibility']:
            # concat the points of lenghts (N1,N2...) to a longer one with length (N1+N2+...)
            ret[key] = np.concatenate(elems, axis=0)
            # record the point numbers for original batches
            ret['batched_model_point_lengths'] = np.array(
                [len(p) for p in elems], dtype=np.int32)
        elif key in ['rand_model_points', ]:
            # concat the points of lenghts (N1,N2...) to a longer one with length (N1+N2+...)
            ret[key] = np.concatenate(elems, axis=0)
            # record the point numbers for original batches
            ret['batched_rand_model_point_lengths'] = np.array(
                [len(p) for p in elems], dtype=np.int32)
        elif key in ['model_point_features']:
            ret[key] = np.concatenate(elems, axis=0)

        # ['odometry/tq','odometry/RT','odometry/invRT' ]:
        elif key in ['image', 'depth', 'K', 'RT', 'original_RT' ,'rand_RT', 'correspondences_2d3d', 'scale',  'POSECNN_RT','rendered_image', 'rendered_depth', 'rendered_RT', '3d_keypoint_inds', '3d_keypoints', 'mask', 'ren_mask']: #'depth_coords2d','lifted_points', 
            try:
                ret[key] = np.stack(elems, axis=0)
            except:
                print(key, flush=True)
                raise
        elif key == 'metrics':
            ret[key] = elems
        else:
            ret[key] = []
            for e in elems:
                ret[key].append(e)

    return ret


def get_correspondences(src_pcd, tgt_pcd, search_voxel_size, K=None, trans=None):
    if trans is not None:
        src_pcd.transform(trans)
    pcd_tree = o3d.geometry.KDTreeFlann(tgt_pcd)

    correspondences = []
    for i, point in enumerate(src_pcd.points):
        [count, idx, _] = pcd_tree.search_radius_vector_3d(
            point, search_voxel_size)
        if K is not None:
            idx = idx[:K]
        for j in idx:
            correspondences.append([i, j])

    correspondences = np.array(correspondences)
    # correspondences = torch.from_numpy(correspondences)
    return correspondences


def to_pcd(xyz):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    return pcd


def to_tsfm(rot, trans):
    tsfm = np.eye(4)
    tsfm[:3, :3] = rot
    tsfm[:3, 3] = trans.flatten()
    return tsfm


def CameraIntrinsicUpdate(old_K, aug_param):
    '''
    old_K: array of shape (N,3,3), the old camera intrinsic parameters
    aug_pram: dict, the data augmentation parameters
    '''
    aug_type = aug_param['aug_type']
    assert aug_type in ['crop', 'scale', 'flip']

    new_K = np.copy(old_K)
    if aug_type == 'crop':
        cx, cy = aug_param['crop/left_top_corner']  # x,y
        new_K[..., 0, 2] = new_K[..., 0, 2] - cx
        new_K[..., 1, 2] = new_K[..., 1, 2] - cy
    elif aug_type == 'scale':
        s_x, s_y = aug_param['scale/scale']
        new_K[..., 0, 0] = s_x * new_K[..., 0, 0]
        new_K[..., 1, 1] = s_y * new_K[..., 1, 1]

        new_K[..., 0, 2] = s_x * new_K[..., 0, 2]
        new_K[..., 1, 2] = s_y * new_K[..., 1, 2]
    elif aug_type == 'flip':
        w = aug_param['flip/width']
        # h = aug_param['flip/heigh']
        new_K[..., 0, 2] = w - new_K[..., 0, 2]  # px' = w-px
        # new_K[...,1,2] = h- new_K[...,1,2]
        new_K[..., 0, 0] = - new_K[..., 0, 0]  # fx' = -fx

    return new_K


def crop_transform(images, depths, Ks, crop_param, ):
    assert(len(images) == len(depths) == len(Ks))

    crop_type = crop_param["crop_type"]
    assert(crop_type in ["fixed", "center", "random"])

    crop_size = crop_param["crop_size"]
    iheight, iwidth = images[0].shape[:2]

    if crop_type == "fixed":
        lt_corner = crop_param["lt_corner"]
        op = transforms.Crop(
            lt_corner[0], lt_corner[1], crop_size[0], crop_size[1])
    elif crop_type == "center":
        op = transforms.CenterCrop(crop_size)

        ci, cj, _, _ = op.get_params(images[0], crop_size)
        lt_corner = ci, cj

    elif crop_type == "random":
        op = transforms.RandomCrop((iheight, iwidth), crop_size)

        lt_corner = op.i, op.j

    for i, _ in enumerate(images):
        images[i] = op(images[i])
        depths[i] = op(depths[i])

        Ks[i] = CameraIntrinsicUpdate(Ks[i],
                                      {"aug_type": "crop", "crop/left_top_corner": (lt_corner[1], lt_corner[0])})

    return images, depths, Ks


# def patch_crop(image, depth, mask, K_old, margin_ratio=0.2, output_size=128, offset_ratio=(0,0),bbox=None, mask_depth=True):
def patch_crop(image, depth, mask, K_old, margin_ratio=0.2, output_size=128, offset_ratio=(0,0),bbox=None, mask_depth=False):
    '''
        image: HxWx3
        mask: HxW
        K_old: 3x3
        offset: (offset_h, offset_w)
    '''

    H, W, _ = image.shape
    
    mask = mask.astype('uint8')*255
    if bbox is None:
        _x, _y, _w, _h = cv2.boundingRect(mask)
    else:
        _x, _y, _w, _h = bbox[1], bbox[0], bbox[3]-bbox[1], bbox[2]-bbox[0]

    # center = [_x+_w/2, _y+_h/2]
    center = [_x+_w/2+offset_ratio[1]*_w, _y+_h/2+offset_ratio[0]*_h ]

    L = int(max(_w, _h) * (1+2*margin_ratio))
    
    if L<0:
        #TODO
        print(mask.sum(), depth.sum(), '!!!', flush=True)
        L=128

    x = max(0, int(center[0] - L/2))
    y = max(0, int(center[1] - L/2))

    crop = image[y:y+L, x:x+L]
    # only keep the ROI depth

    if mask_depth:
        depth[mask < 1] = 0 # removed by dy at 0810
    depth_crop = depth[y:y+L, x:x+L]
    mask_crop = mask[y:y+L, x:x+L]

    

    # w=h=int ((1+2*margin_ratio)*L) # actual crop size
    w = h = L  # actual crop size
    # automatically handle the "out of range" problem
    patch = np.zeros([h, w, 3], dtype=image.dtype)
    # depth_patch = np.ones([h, w], dtype=depth.dtype)
    depth_patch = np.zeros([h, w], dtype=depth.dtype)
    mask_patch = np.zeros([h, w], dtype=depth.dtype)

    try:
        xp = 0
        yp = 0
        patch[xp: xp+crop.shape[0], yp:yp+crop.shape[1]] = crop
        depth_patch[xp: xp+crop.shape[0], yp:yp+crop.shape[1]] = depth_crop
        mask_patch[xp: xp+crop.shape[0], yp:yp+crop.shape[1]] = mask_crop
    except:
        import pdb
        pdb.set_trace()
    patch = cv2.resize(patch, (output_size, output_size),
                       interpolation=cv2.INTER_LINEAR)
    depth_patch = cv2.resize(
        depth_patch, (output_size, output_size), interpolation=cv2.INTER_NEAREST)
    mask_patch = cv2.resize(
        mask_patch, (output_size, output_size), interpolation=cv2.INTER_NEAREST)

    # update the intrinsic parameters
    K_new = np.zeros_like(K_old)
    scale = output_size/L
    K_new[0, 2] = (K_old[0, 2]-x)*scale
    K_new[1, 2] = (K_old[1, 2]-y)*scale
    K_new[0, 0] = K_old[0, 0]*scale
    K_new[1, 1] = K_old[1, 1]*scale
    K_new[2, 2] = 1

    # return patch, depth_patch, K_new
    return patch, depth_patch, mask_patch, K_new


def preprocess_deepim(
    input_dict,
    max_points,
    correspondence_radius,
    normalize_model=True,
    rand_transform_model=False,  # False,#True,
    rand_rgb_transformer=None,
    image_scale=None,
    patch_cropper=None,  # func patch_crop(...)
    
):
    output_dict = copy.deepcopy(input_dict)

    ################################### process 3D point clouds ###################################

    if (output_dict['model_points'].shape[0] > max_points):
        # if(output_dict['model_points'].shape[0] > 20000):
        idx = np.random.permutation(
            output_dict['model_points'].shape[0])[:max_points]
        print(idx, output_dict['model_points'].shape, flush=True)
        output_dict['model_points'] = output_dict['model_points'][idx]
        output_dict['model_point_features'] = output_dict['model_point_features'][idx]

    output_dict['original_RT'] = copy.deepcopy(output_dict['RT'])
    if normalize_model:
        points = output_dict['model_points']
        mean = points.mean(axis=0)
        scope = points.max(axis=0)-points.min(axis=0)
        points_normalize = (points-mean)/scope.max()
        # points_normalize.tofile(bin_save_path)
        # modify the extrinsic parameters
        output_dict['RT'][:, 3:] = output_dict['RT'][:, :3] @ mean[:,
                                                                   None] + output_dict['RT'][:, 3:]  # 3x3 @ 3x1 + 3x1
        # input_dict['RT'][:,:3] *=scope.max()
        output_dict['scale'] = scope.max()
        output_dict['original_model_points'] = output_dict['model_points']
        output_dict['model_points'] = points_normalize


    if rand_transform_model:
        points = output_dict['model_points']
        rand_quat = np.random.randn(1, 4)
        rand_quat = rand_quat/np.linalg.norm(rand_quat, axis=-1)
        rand_rot = quaternion.as_rotation_matrix(
            quaternion.from_float_array(rand_quat)).squeeze()  # 3x3
        output_dict['rand_model_points'] = (
            rand_rot@ points.T).T.astype(np.float32)
        output_dict['rand_RT'] = copy.deepcopy(output_dict['RT'])
        # output_dict['RT'][:,:3]@rand_rot.T
        output_dict['rand_RT'][:, :3] = rand_rot
        output_dict['rand_RT'][:, 3] = 0

    ################################### process 2D images ###################################
    # carve out image patches
    if patch_cropper is not None:
        ref_mask = output_dict['depth'] > 0
        output_dict['image'], output_dict['depth'], output_dict['K'] = patch_cropper(
            output_dict['image'], output_dict['depth'],  ref_mask, output_dict['K'])

        output_dict['rendered_image'], output_dict['rendered_depth'], _ = patch_cropper(
            output_dict['rendered_image'], output_dict['rendered_depth'],  ref_mask, output_dict['K'].copy() )

    # rescale image
    if image_scale is not None:
        output_dict['image'] = cv2.resize(output_dict['image'],
                                          (output_dict['image'].shape[1]*image_scale,
                                           output_dict['image'].shape[0]*image_scale),
                                          interpolation=cv2.INTER_AREA)
        output_dict['depth'] = cv2.resize(output_dict['depth'],
                                          (output_dict['depth'].shape[1]*image_scale,
                                           output_dict['depth'].shape[0]*image_scale),
                                          interpolation=cv2.INTER_NEAREST)
        output_dict['K'][:2] = output_dict['K'][:2]*image_scale

    # lift depth
    depth = output_dict['depth'].squeeze()  # H,W
    depth_pts, depth_coords2d = mask_depth_to_point_cloud(
        depth != 0, depth, output_dict['K'])
    depth_pts = (output_dict['RT'][:, :3].T@(depth_pts.T - output_dict['RT']
                                             [:, 3:])).T / output_dict['scale']  # transformed to the model frame

    # find 2d-3d correspondences
    tsfm = np.eye(4)
    tsfm[:3] = output_dict['RT']
    model_pcd = output_dict['model_points']

    correspondences_2d3d = get_correspondences(
        to_pcd(depth_pts), to_pcd(model_pcd),  correspondence_radius, K=5)
    if len(correspondences_2d3d.shape) < 2 or len(correspondences_2d3d) < 10:
        print(depth_pts.shape, model_pcd.shape)
        print("correspondences_2d3d.shape:",
              correspondences_2d3d.shape, flush=True)
        # raise ValueError("Too few correspondences are found!")
        raise Exception("Too few correspondences are found!")

    output_dict['depth_coords2d'] = depth_coords2d
    output_dict['lifted_points'] = depth_pts
    # output_dict['correspondences_2d3d'] = np.zeros(1)#correspondences_2d3d
    output_dict['correspondences_2d3d'] = correspondences_2d3d

    if rand_rgb_transformer is not None:
        output_dict['image'], _, _ = rand_rgb_transformer(output_dict['image'])
    # TO TENSOR
    output_dict['image'] = (output_dict['image'].astype(
        np.float32)/255.0).transpose([2, 0, 1])  # .mean(axis=0, keepdims=True) # 1,H,W
    output_dict['depth'] = output_dict['depth'].astype(np.float32)[
        None]  # 1,H,W

    return output_dict

def preprocess(
    input_dict,
    max_points,
    correspondence_radius,
    normalize_model=True,
    rand_transform_model=False, 
    rand_rgb_transformer=None,
    image_scale=None,
    crop_param=None,
    kp_3d_param=None,
    use_coords_as_3d_feat=False,
    find_2d3d_correspondence=True,
    
):
    output_dict = copy.deepcopy(input_dict)

    ################################### process 3D point clouds ###################################
    if use_coords_as_3d_feat:
        output_dict['model_point_features'] = output_dict['model_points'][:,:3]

    if (output_dict['model_points'].shape[0] > max_points):
        # if(output_dict['model_points'].shape[0] > 20000):
        idx = np.random.permutation(
            output_dict['model_points'].shape[0])[:max_points]
        print(idx, output_dict['model_points'].shape, flush=True)
        output_dict['model_points'] = output_dict['model_points'][idx]
        output_dict['model_point_features'] = output_dict['model_point_features'][idx]

    output_dict['original_RT'] = copy.deepcopy(output_dict['RT'])
    output_dict['original_model_points'] = output_dict['model_points']
    if normalize_model:
        points = output_dict['model_points']
        mean = points.mean(axis=0)
        scope = points.max(axis=0)-points.min(axis=0)
        points_normalize = (points-mean)/scope.max()
        # modify the extrinsic parameters
        output_dict['RT'][:, 3:] = output_dict['RT'][:, :3] @ mean[:,
                                                                   None] + output_dict['RT'][:, 3:]  # 3x3 @ 3x1 + 3x1
        output_dict['scale'] = scope.max()
        output_dict['model_points'] = points_normalize


    if rand_transform_model:
        points = output_dict['model_points']
        rand_quat = np.random.randn(1, 4)
        rand_quat = rand_quat/np.linalg.norm(rand_quat, axis=-1)
        rand_rot = quaternion.as_rotation_matrix(
            quaternion.from_float_array(rand_quat)).squeeze()  # 3x3
        output_dict['rand_model_points'] = (
            rand_rot@ points.T).T.astype(np.float32)
        output_dict['rand_RT'] = copy.deepcopy(output_dict['RT'])
        output_dict['rand_RT'][:, :3] = rand_rot
        output_dict['rand_RT'][:, 3] = 0

    ################################### process 2D images ###################################
    #crop image
    if crop_param is not None:# and output_dict['mask'].sum()>0:
        #without random cropping
        if not crop_param.rand_crop: 
            if crop_param.get("crop_with_init_pose", False):
                # bbox= output_dict.get('bbox', None)
                bbox=None
                output_dict['image'], output_dict['depth'], output_dict['mask'], output_dict['K'] = patch_crop(output_dict['image'], output_dict['depth'], mask=output_dict['ren_mask'],
                                K_old=output_dict['K'], margin_ratio=crop_param.margin_ratio, output_size=crop_param.output_size,  bbox=bbox
                                                )
            elif crop_param.get("crop_with_rand_bbox_shift", True): 
                bbox= output_dict.get('bbox', None)
                # offset_ratio= [truncated_normal(0,0.5,-1,1)*crop_param.max_rand_offset_ratio, truncated_normal(0,0.5,-1,1)*crop_param.max_rand_offset_ratio] 
                offset_ratio= [truncated_normal(0,0.33,-1,1)*1, truncated_normal(0,0.33,-1,1)*1] 
                output_dict['image'], output_dict['depth'], output_dict['mask'], output_dict['K'] = patch_crop(output_dict['image'], output_dict['depth'], mask=output_dict['mask'],
                                                    K_old=output_dict['K'], margin_ratio=crop_param.margin_ratio, output_size=crop_param.output_size, offset_ratio=offset_ratio, bbox=output_dict.get('bbox', None) 
                                                )
            else:
                bbox= output_dict.get('bbox', None)
                output_dict['image'], output_dict['depth'], output_dict['mask'], output_dict['K'] = patch_crop(output_dict['image'], output_dict['depth'], mask=output_dict['mask'],
                                                    K_old=output_dict['K'], margin_ratio=crop_param.margin_ratio, output_size=crop_param.output_size, bbox=output_dict.get('bbox', None) 
                                                )
        else:
            margin_ratio= truncated_normal(0.5, 0.5, 0, 1) *crop_param.max_rand_margin_ratio
            offset_ratio= [truncated_normal(0,0.5,-1,1)*crop_param.max_rand_offset_ratio, truncated_normal(0,0.5,-1,1)*crop_param.max_rand_offset_ratio] 
            output_dict['image'], output_dict['depth'], output_dict['mask'], output_dict['K'] = patch_crop(output_dict['image'], output_dict['depth'], mask=output_dict['mask'],
                                                K_old=output_dict['K'], margin_ratio=margin_ratio, output_size=crop_param.output_size, offset_ratio=offset_ratio,  bbox=output_dict.get('bbox', None) 
                                              )
            
    # rescale image
    if image_scale is not None:
        output_dict['image'] = cv2.resize(output_dict['image'],
                                          (output_dict['image'].shape[1]*image_scale,
                                           output_dict['image'].shape[0]*image_scale),
                                          interpolation=cv2.INTER_AREA)
        output_dict['depth'] = cv2.resize(output_dict['depth'],
                                          (output_dict['depth'].shape[1]*image_scale,
                                           output_dict['depth'].shape[0]*image_scale),
                                          interpolation=cv2.INTER_NEAREST)
        output_dict['K'][:2] = output_dict['K'][:2]*image_scale

    # lift depths
    depth = output_dict['depth'].squeeze()  # H,W
    depth_pts, depth_coords2d = mask_depth_to_point_cloud(
        depth != 0, depth, output_dict['K'])

    depth_pts = (output_dict['RT'][:, :3].T@(depth_pts.T - output_dict['RT']
                                             [:, 3:])).T / output_dict['scale']  # transformed to the model frame

    # find 2d-3d correspondences
    if find_2d3d_correspondence:
        tsfm = np.eye(4)
        tsfm[:3] = output_dict['RT']
        model_pcd = output_dict['model_points']
        correspondences_2d3d = get_correspondences(
            to_pcd(depth_pts), to_pcd(model_pcd),  correspondence_radius, K=5)
        if len(correspondences_2d3d.shape) < 2 or len(correspondences_2d3d) < 10:# or ( "mask" in output_dict and output_dict['mask'].sum()<10 ) :
            print(depth_pts.shape, model_pcd.shape)
            print("correspondences_2d3d.shape:",
                  correspondences_2d3d.shape, flush=True)
            raise Exception("Too few correspondences are found!")

        output_dict['depth_coords2d'] = depth_coords2d
        output_dict['lifted_points'] = depth_pts
        output_dict['correspondences_2d3d'] = correspondences_2d3d
    else:
        output_dict['depth_coords2d'] = depth_coords2d
        output_dict['lifted_points'] = depth_pts
        output_dict['correspondences_2d3d'] = np.zeros([10,2], dtype=np.int64) 


    if rand_rgb_transformer is not None:
        output_dict['image'], _, _ = rand_rgb_transformer(output_dict['image'])
    # TO TENSORs
    output_dict['image'] = (output_dict['image'].astype(
        np.float32)/255.0).transpose([2, 0, 1])  # .mean(axis=0, keepdims=True) # 1,H,W
    output_dict['depth'] = output_dict['depth'].astype(np.float32)[
        None]  # 1,H,W

    return output_dict

def batch_grid_subsampling_kpconv(points, batches_len, features=None, labels=None, sampleDl=0.1, max_p=0, verbose=0, random_grid_orient=True):
    """
    CPP wrapper for a grid subsampling (method = barycenter for points and features)
    """
    if (features is None) and (labels is None):
        s_points, s_len = cpp_subsampling.subsample_batch(points,
                                                          batches_len,
                                                          sampleDl=sampleDl,
                                                          max_p=max_p,
                                                          verbose=verbose)
        return torch.from_numpy(s_points), torch.from_numpy(s_len)

    elif (labels is None):
        s_points, s_len, s_features = cpp_subsampling.subsample_batch(points,
                                                                      batches_len,
                                                                      features=features,
                                                                      sampleDl=sampleDl,
                                                                      max_p=max_p,
                                                                      verbose=verbose)
        return torch.from_numpy(s_points), torch.from_numpy(s_len), torch.from_numpy(s_features)

    elif (features is None):
        s_points, s_len, s_labels = cpp_subsampling.subsample_batch(points,
                                                                    batches_len,
                                                                    classes=labels,
                                                                    sampleDl=sampleDl,
                                                                    max_p=max_p,
                                                                    verbose=verbose)
        return torch.from_numpy(s_points), torch.from_numpy(s_len), torch.from_numpy(s_labels)

    else:
        s_points, s_len, s_features, s_labels = cpp_subsampling.subsample_batch(points,
                                                                                batches_len,
                                                                                features=features,
                                                                                classes=labels,
                                                                                sampleDl=sampleDl,
                                                                                max_p=max_p,
                                                                                verbose=verbose)
        return torch.from_numpy(s_points), torch.from_numpy(s_len), torch.from_numpy(s_features), torch.from_numpy(s_labels)


def batch_neighbors_kpconv(queries, supports, q_batches, s_batches, radius, max_neighbors):
    """
    Computes neighbors for a batch of queries and supports, apply radius search
    :param queries: (N1, 3) the query points
    :param supports: (N2, 3) the support points
    :param q_batches: (B) the list of lengths of batch elements in queries
    :param s_batches: (B)the list of lengths of batch elements in supports
    :param radius: float32
    :return: neighbors indices
    """

    neighbors = cpp_neighbors.batch_query(
        queries, supports, q_batches, s_batches, radius=radius)
    # print("neighbors.shape" , neighbors.shape, queries.shape,flush=True)
    if max_neighbors > 0:
        return torch.from_numpy(neighbors[:, :max_neighbors])
    else:
        return torch.from_numpy(neighbors)


def collate_fn_descriptor(list_data, config, neighborhood_limits):
    ret = merge_batch(list_data)

    batched_points = torch.from_numpy(ret['model_points'])
    batched_lengths = torch.from_numpy(ret['batched_model_point_lengths'])
    batched_features = torch.from_numpy(ret['model_point_features'])

    if ret.get('rand_model_points', None) is not None:
        batched_rand_points = torch.from_numpy(ret['rand_model_points'])
        batched_rand_lengths = torch.from_numpy(
            ret['batched_rand_model_point_lengths'])

        batched_points = torch.cat(
            [batched_points, batched_rand_points], dim=0)
        batched_lengths = torch.cat(
            [batched_lengths, batched_rand_lengths], dim=0)
        batched_features = torch.cat(
            [batched_features, batched_features], dim=0)

    # Starting radius of convolutions
    r_normal = config.first_subsampling_dl * config.conv_radius
    # Starting layer
    layer_blocks = []
    layer = 0

    # Lists of inputs
    input_points = []
    input_neighbors = []
    input_pools = []
    input_upsamples = []
    input_batches_len = []
    timer = Timer()
    for block_i, block in enumerate(config.architecture):
        # Stop when meeting a global pooling or upsampling
        if 'global' in block or 'upsample' in block:
            break

        # Get all blocks of the layer
        if not ('pool' in block or 'strided' in block):
            layer_blocks += [block]
            if block_i < len(config.architecture) - 1 and not ('upsample' in config.architecture[block_i + 1]):
                continue

        # Convolution neighbors indices
        # *****************************

        if layer_blocks:
            # Convolutions are done in this layer, compute the neighbors with the good radius
            if np.any(['deformable' in blck for blck in layer_blocks[:-1]]):
                r = r_normal * config.deform_radius / config.conv_radius
            else:
                r = r_normal
            conv_i = batch_neighbors_kpconv(
                batched_points, batched_points, batched_lengths, batched_lengths, r, neighborhood_limits[layer])

        else:
            # This layer only perform pooling, no neighbors required
            conv_i = torch.zeros((0, 1), dtype=torch.int64)

        # Pooling neighbors indices
        # *************************

        if 'pool' in block or 'strided' in block:

            # New subsampling length
            dl = 2 * r_normal / config.conv_radius

            # Subsampled points
            pool_p, pool_b = batch_grid_subsampling_kpconv(
                batched_points, batched_lengths, sampleDl=dl)

            # Radius of pooled neighbors
            if 'deformable' in block:
                r = r_normal * config.deform_radius / config.conv_radius
            else:
                r = r_normal

            # Subsample indices
            pool_i = batch_neighbors_kpconv(
                pool_p, batched_points, pool_b, batched_lengths, r, neighborhood_limits[layer])

            # Upsample indices (with the radius of the next layer to keep wanted density)
            up_i = batch_neighbors_kpconv(
                batched_points, pool_p, batched_lengths, pool_b, 2 * r, neighborhood_limits[layer])

        else:
            # No pooling in the end of this layer, no pooling indices required
            pool_i = torch.zeros((0, 1), dtype=torch.int64)
            pool_p = torch.zeros((0, 3), dtype=torch.float32)
            pool_b = torch.zeros((0,), dtype=torch.int64)
            up_i = torch.zeros((0, 1), dtype=torch.int64)

        # Updating input lists
        input_points += [batched_points.float()]
        input_neighbors += [conv_i.long()]
        input_pools += [pool_i.long()]
        input_upsamples += [up_i.long()]
        input_batches_len += [batched_lengths]

        # New points for next layer
        batched_points = pool_p
        batched_lengths = pool_b

        # Update radius and reset blocks
        r_normal *= 2
        layer += 1
        layer_blocks = []

    ###############
    # Return inputs
    ###############
    dict_inputs = {
        "idx": ret["idx"],
        'model_points': input_points,
        'visibility': torch.from_numpy(ret['visibility']),
        'neighbors': input_neighbors,
        'pools': input_pools,
        'upsamples': input_upsamples,
        'model_point_features': batched_features.float(),
        'stack_lengths': input_batches_len,
        'image': torch.from_numpy(ret['image']),
        'depth': torch.from_numpy(ret['depth']),
        'mask': torch.from_numpy(ret['mask']),
        'ren_mask': torch.from_numpy(ret['ren_mask']),
        'K': torch.from_numpy(ret['K']),
        'RT': torch.from_numpy(ret['RT']),
        'original_RT': torch.from_numpy(ret['original_RT']),
        'POSECNN_RT': torch.from_numpy(ret.get('POSECNN_RT', np.zeros_like(ret['RT']) ) ),
        'rand_RT': torch.from_numpy(ret.get('rand_RT', np.zeros_like(ret['RT']))),
        # "lifted_points": torch.from_numpy(ret['lifted_points']),
        "lifted_points": [torch.from_numpy(d) for d in ret['lifted_points'] ] ,
        # 'depth_coords2d': torch.from_numpy(ret['depth_coords2d']),
        'depth_coords2d': [torch.from_numpy(d) for d in ret['depth_coords2d']],
        "correspondences_2d3d": torch.from_numpy(ret['correspondences_2d3d']),
        "original_model_points": torch.from_numpy(ret['original_model_points']),
        "class_name": ret['class_name'],
        "3d_keypoint_inds": torch.from_numpy(ret['3d_keypoint_inds']),
        "3d_keypoints": torch.from_numpy(ret['3d_keypoints'] ) 
    }

    return dict_inputs


def collate_fn_descriptor_deepim(list_data, config, neighborhood_limits):
    ret = merge_batch(list_data)

    batched_points = torch.from_numpy(ret['model_points'])
    batched_lengths = torch.from_numpy(ret['batched_model_point_lengths'])
    batched_features = torch.from_numpy(ret['model_point_features'])
    

    if ret.get('rand_model_points', None) is not None:
        # torch.from_numpy(np.concatenate(batched_points_list, axis=0))
        batched_rand_points = torch.from_numpy(ret['rand_model_points'])
        # torch.from_numpy(np.concatenate(batched_points_list, axis=0))
        batched_rand_lengths = torch.from_numpy(
            ret['batched_rand_model_point_lengths'])

        batched_points = torch.cat(
            [batched_points, batched_rand_points], dim=0)
        batched_lengths = torch.cat(
            [batched_lengths, batched_rand_lengths], dim=0)
        batched_features = torch.cat(
            [batched_features, batched_features], dim=0)

    # Starting radius of convolutions
    r_normal = config.first_subsampling_dl * config.conv_radius
    # Starting layer
    layer_blocks = []
    layer = 0

    # Lists of inputs
    input_points = []
    input_neighbors = []
    input_pools = []
    input_upsamples = []
    input_batches_len = []
    timer = Timer()
    for block_i, block in enumerate(config.architecture):
        # timer.tic()

        # Stop when meeting a global pooling or upsampling
        if 'global' in block or 'upsample' in block:
            break

        # Get all blocks of the layer
        if not ('pool' in block or 'strided' in block):
            layer_blocks += [block]
            if block_i < len(config.architecture) - 1 and not ('upsample' in config.architecture[block_i + 1]):
                continue

        # Convolution neighbors indices
        # *****************************

        if layer_blocks:
            # Convolutions are done in this layer, compute the neighbors with the good radius
            if np.any(['deformable' in blck for blck in layer_blocks[:-1]]):
                r = r_normal * config.deform_radius / config.conv_radius
            else:
                r = r_normal
            conv_i = batch_neighbors_kpconv(
                batched_points, batched_points, batched_lengths, batched_lengths, r, neighborhood_limits[layer])

        else:
            # This layer only perform pooling, no neighbors required
            conv_i = torch.zeros((0, 1), dtype=torch.int64)

        # Pooling neighbors indices
        # *************************

        # If end of layer is a pooling operation
        if 'pool' in block or 'strided' in block:

            # New subsampling length
            dl = 2 * r_normal / config.conv_radius

            # Subsampled points
            pool_p, pool_b = batch_grid_subsampling_kpconv(
                batched_points, batched_lengths, sampleDl=dl)

            # Radius of pooled neighbors
            if 'deformable' in block:
                r = r_normal * config.deform_radius / config.conv_radius
            else:
                r = r_normal

            # Subsample indices
            pool_i = batch_neighbors_kpconv(
                pool_p, batched_points, pool_b, batched_lengths, r, neighborhood_limits[layer])

            # Upsample indices (with the radius of the next layer to keep wanted density)
            up_i = batch_neighbors_kpconv(
                batched_points, pool_p, batched_lengths, pool_b, 2 * r, neighborhood_limits[layer])

        else:
            # No pooling in the end of this layer, no pooling indices required
            pool_i = torch.zeros((0, 1), dtype=torch.int64)
            pool_p = torch.zeros((0, 3), dtype=torch.float32)
            pool_b = torch.zeros((0,), dtype=torch.int64)
            up_i = torch.zeros((0, 1), dtype=torch.int64)

        # Updating input lists
        input_points += [batched_points.float()]
        input_neighbors += [conv_i.long()]
        input_pools += [pool_i.long()]
        input_upsamples += [up_i.long()]
        input_batches_len += [batched_lengths]

        # New points for next layer
        batched_points = pool_p
        batched_lengths = pool_b

        # Update radius and reset blocks
        r_normal *= 2
        layer += 1
        layer_blocks = []

        # timer.toc()
    ###############
    # Return inputs
    ###############
    dict_inputs = {
        "idx": ret["idx"],
        'model_points': input_points,
        'visibility': torch.from_numpy(ret['visibility']),
        'neighbors': input_neighbors,
        'pools': input_pools,
        'upsamples': input_upsamples,
        'model_point_features': batched_features.float(),
        'stack_lengths': input_batches_len,
        'image': torch.from_numpy(ret['image']),
        'depth': torch.from_numpy(ret['depth']),
        "ren_mask": torch.from_numpy(ret['ren_mask']),
        'K': torch.from_numpy(ret['K']),
        'RT': torch.from_numpy(ret['RT']),
        'original_RT': torch.from_numpy(ret['original_RT']),
        'rendered_RT': torch.from_numpy(ret['rendered_RT']) if ret.get('rendered_RT', None) is not None else None ,
        'POSECNN_RT': torch.from_numpy(ret.get('POSECNN_RT', np.zeros_like(ret['RT']) ) ),
        # TODO
        'rand_RT': torch.from_numpy(ret.get('rand_RT', np.zeros_like(ret['RT']))),
        # "lifted_points": torch.from_numpy(ret['lifted_points']),
        "lifted_points": [torch.from_numpy(d) for d in ret['lifted_points'] ] ,
        # 'depth_coords2d': torch.from_numpy(ret['depth_coords2d']),
        'depth_coords2d': [torch.from_numpy(d) for d in ret['depth_coords2d']],
        "correspondences_2d3d": torch.from_numpy(ret['correspondences_2d3d']),
        "original_model_points": torch.from_numpy(ret['original_model_points']),
        "class_name": ret['class_name'],
    }

    return dict_inputs


def calibrate_neighbors(dataset, config, collate_fn, keep_ratio=0.8, samples_threshold=2000):
    timer = Timer()
    last_display = timer.total_time

    # From config parameter, compute higher bound of neighbors number in a neighborhood
    hist_n = int(np.ceil(4 / 3 * np.pi * (config.deform_radius + 1) ** 3))
    neighb_hists = np.zeros((config.num_layers, hist_n), dtype=np.int32)

    # Get histogram of neighborhood sizes i in 1 epoch max.
    for i in range(len(dataset)):
        timer.tic()

        batched_input = collate_fn(
            [dataset[i]], config, neighborhood_limits=[hist_n] * 5)
        # update histogram
        counts = [torch.sum(neighb_mat < neighb_mat.shape[0], dim=1).numpy()
                  for neighb_mat in batched_input['neighbors']]
        
        hists = [np.bincount(c, minlength=hist_n)[:hist_n] for c in counts]
        neighb_hists += np.vstack(hists)
        timer.toc()

        if timer.total_time - last_display > 0.1:
            last_display = timer.total_time
            print(f"Calib Neighbors {i:08d}: timings {timer.total_time:4.2f}s")

        if np.min(np.sum(neighb_hists, axis=1)) > samples_threshold:
            break

    cumsum = np.cumsum(neighb_hists.T, axis=0)
    percentiles = np.sum(cumsum < (keep_ratio * cumsum[hist_n - 1, :]), axis=0)

    neighborhood_limits = percentiles
    print('\n')

    return neighborhood_limits


def get_dataloader(dataset, kpconv_config, batch_size=1, num_workers=4, shuffle=True, sampler=None, neighborhood_limits=None):
    if neighborhood_limits is None:
        # neighborhood_limits = calibrate_neighbors(dataset, dataset.config, collate_fn=collate_fn_descriptor)
        neighborhood_limits = calibrate_neighbors(
            dataset, kpconv_config, collate_fn=collate_fn_descriptor)
    print("neighborhood:", neighborhood_limits)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        # https://discuss.pytorch.org/t/supplying-arguments-to-collate-fn/25754/4
        collate_fn=partial(collate_fn_descriptor, config=kpconv_config,
                           neighborhood_limits=neighborhood_limits),
        sampler=sampler,
        drop_last=False
    )
    return dataloader, neighborhood_limits

def get_dataloader_deepim(dataset, kpconv_config, batch_size=1, num_workers=4, shuffle=True, sampler=None, neighborhood_limits=None):
    if neighborhood_limits is None:
        neighborhood_limits = calibrate_neighbors(
            dataset, kpconv_config, collate_fn=collate_fn_descriptor_deepim)
    print("Neighborhood:", neighborhood_limits)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        # https://discuss.pytorch.org/t/supplying-arguments-to-collate-fn/25754/4
        collate_fn=partial(collate_fn_descriptor_deepim, config=kpconv_config,
                           neighborhood_limits=neighborhood_limits),
        sampler=sampler,
        drop_last=False
    )
    return dataloader, neighborhood_limits

if __name__ == '__main__':
    pass

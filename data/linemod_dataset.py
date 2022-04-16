import numpy as np 
import random
import os 
from data.dataset import Dataset, register_dataset
import pickle
import PIL
import cv2
import torch
import time
import scipy

from utils.geometric import range_to_depth, render_pointcloud
from .transforms import make_transforms 
from thirdparty.kpconv.lib.utils import square_distance
from utils.geometric import rotation_angle
# from utils.visualize import *
from transforms3d.quaternions import mat2quat, quat2mat, qmult
from transforms3d.euler import mat2euler, euler2mat, euler2quat, quat2euler
import math
from config.default import get_cfg

CURRENT_DIR=os.path.dirname(os.path.abspath(__file__))

try:
    from pytorch3d.io import load_obj, load_ply
except:
    print("Warning: error occurs when importing pytorch3d ")
    pass


def se3_q2m(se3_q):
    assert se3_q.size == 7
    se3_mx = np.zeros((3, 4))
    # quat = se3_q[0:4] / LA.norm(se3_q[0:4])
    quat = se3_q[:4]
    R = quat2mat(quat)
    se3_mx[:, :3] = R
    se3_mx[:, 3] = se3_q[4:]
    return se3_mx

def info_convertor(info,):
    """
        [Transform the original kitti info file]
    """

    seqs = info.keys() #['cat']#
    seq_lengths = [len(info[i]) for i in seqs]
    data = []
    for seq in seqs:
        print(seq)
        data.append(info[seq])

    new_infos = {
        "seqs": list(seqs),
        "seq_lengths": seq_lengths,
        "data": data
    }
    return new_infos

def resize(im, target_size, max_size, stride=0, interpolation=cv2.INTER_LINEAR):
    """
    only resize input image to target size and return scale
    :param im: BGR image input by opencv
    :param target_size: one dimensional size (the short side)
    :param max_size: one dimensional max size (the long side)
    :param stride: if given, pad the image to designated stride
    :param interpolation: if given, using given interpolation method to resize image
    :return:
    """
    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(target_size) / float(im_size_min)
    # prevent bigger axis from being more than max_size:
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)
    im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale, interpolation=interpolation)

    if stride == 0:
        return im, im_scale
    else:
        # pad to product of stride
        im_height = int(np.ceil(im.shape[0] / float(stride)) * stride)
        im_width = int(np.ceil(im.shape[1] / float(stride)) * stride)
        im_channel = im.shape[2]
        padded_im = np.zeros((im_height, im_width, im_channel))
        padded_im[: im.shape[0], : im.shape[1], :] = im
        return padded_im, im_scale
def sample_poses(pose_tgt):
    SYN_STD_ROTATION = 15
    SYN_STD_TRANSLATION = 0.01
    ANGLE_MAX=45
    pose_src = pose_tgt.copy()
    num = pose_tgt.shape[0]
    for i in range(num):
        euler = mat2euler(pose_tgt[i, :3, :3])
        euler += SYN_STD_ROTATION * np.random.randn(3) * math.pi / 180.0
        pose_src[i, :3, :3] = euler2mat(euler[0], euler[1], euler[2])

        pose_src[i, 0, 3] = pose_tgt[i, 0, 3]+ SYN_STD_TRANSLATION * np.random.randn(1)
        pose_src[i, 1, 3] = pose_tgt[i, 1, 3] + SYN_STD_TRANSLATION * np.random.randn(1)
        pose_src[i, 2, 3] = pose_tgt[i, 2, 3]  + 5 * SYN_STD_TRANSLATION * np.random.randn(1)

        r_dist = np.arccos((np.trace(pose_src[i, :3,:3].transpose(-1,-2) @ pose_tgt[i, :3,:3]) - 1 )/2)/math.pi*180

        while r_dist > ANGLE_MAX:#or not (16 < center_x < (640 - 16) and 16 < center_y < (480 - 16)):
            # print("r_dist > ANGLE_MAX, resampling...")
            print("Too large angular differences. Resample the pose...")
            euler = mat2euler(pose_tgt[i, :3, :3])
            euler += SYN_STD_ROTATION * np.random.randn(3) * math.pi / 180.0
            pose_src[i, :3, :3] = euler2mat(euler[0], euler[1], euler[2])

            pose_src[i, 0, 3] = pose_tgt[i, 0, 3]+ SYN_STD_TRANSLATION * np.random.randn(1)
            pose_src[i, 1, 3] = pose_tgt[i, 1, 3] + SYN_STD_TRANSLATION * np.random.randn(1)
            pose_src[i, 2, 3] = pose_tgt[i, 2, 3]  + 5 * SYN_STD_TRANSLATION * np.random.randn(1)

            r_dist = np.arccos((np.trace(pose_src[i, :3,:3].transpose(-1,-2) @ pose_tgt[i, :3,:3]) - 1 )/2)*math.pi/180
    return pose_src.squeeze()




@register_dataset
class LinemodDeepIMSynRealV2(Dataset):
    # use deepim 3d model for geometric feature extraction, mingle the synthetic and real data  
    def __init__(self, root_path,
                 info_path, model_point_dim,
                 is_train,
                 prep_func=None,
                 seq_names=None, 
                 cfg={}
                 ):
        super().__init__()

        assert info_path is not None
        assert isinstance(root_path, (tuple, list)) and isinstance(info_path, (tuple, list))
        assert len(root_path) == len(info_path)
        print("Info:",info_path)
        # assert split in ['train', 'val', 'test']
        self.is_train = is_train
        self.VOC_ROOT = get_cfg('DATA').VOC_ROOT#"/DATA/yxu/LINEMOD_DEEPIM/"

        infos=[]
        for ipath in info_path:
            with open(ipath, 'rb') as f:
                info = pickle.load(f)

                if seq_names is not None:
                    for k in list(info.keys()):
                        if k not in seq_names:
                            del info[k]
                infos.append( info_convertor(info) )

        #merge multiple infos 
        self.infos = infos[0]
        self.infos['dataset_idx'] = [0]*len(self.infos['seqs'])
        for i, info in enumerate(infos[1:]):
            for k in self.infos:
                if k == 'dataset_idx':
                    self.infos[k].extend([i+1]*len(info['seqs']))
                else:
                    self.infos[k].extend(info[k])


        self.root_paths = root_path
        self.model_point_dim = model_point_dim
        # self.max_points=max_points#30000
        self.prep_func=prep_func
        # self.rgb_transformer = None #make_transforms(None, is_train=is_train)
        self.rgb_transformer = make_transforms(None, is_train=is_train)
        print("dataset size:",self.__len__())

        self.init_pose_type = cfg.get("init_post_type", "POSECNN_LINEMOD" ) 
        # self.init_pose_type = cfg.get("init_post_type", "PVNET_LINEMOD_OCC" ) 
#         self.init_pose_type = cfg.get("init_post_type", "PVNET_LINEMOD" ) 
        print("INIT_POSE_TYPE:", self.init_pose_type)
        #Load posecnn results
        if not self.is_train:
            with open(f"{CURRENT_DIR}/../EXPDATA/init_poses/linemod_posecnn_results.pkl", 'rb') as f:
                self.pose_cnn_results_test_posecnn=pickle.load(f)
            try:
                if self.init_pose_type == "POSECNN_LINEMOD":
                    #load posecnn results 
                    self.pose_cnn_results_test=self.pose_cnn_results_test_posecnn
                elif self.init_pose_type =="PVNET_LINEMOD":
                    self.pose_cnn_results_test=np.load(f"{CURRENT_DIR}/../EXPDATA/init_poses/pvnet/pvnet_linemod_test.npy", allow_pickle=True).flat[0]
                elif self.init_pose_type =="PVNET_LINEMOD_OCC":
                    self.pose_cnn_results_test=np.load(f"{CURRENT_DIR}/../EXPDATA/init_poses/pvnet/pvnet_linemodocc_test.npy", allow_pickle=True).flat[0]
                else: 
                    raise NotImplementedError
            except:
                print("Loading posecnn results failed!")
                self.pose_cnn_results_test=None
            try:
                # self.blender_to_bop_pose=np.load(f"{CURRENT_DIR}/../EXPDATA/init_poses/metricpose/blender2bop_RT.npy", allow_pickle=True).flat[0]
                self.blender_to_bop_pose=np.load(f"{CURRENT_DIR}/../EXPDATA/init_poses/pose_conversion/blender2bop_RT.npy", allow_pickle=True).flat[0]
            except:
                print("Loading pose conversion matrix failed!")
                self.blender_to_bop_pose=None 
                
        else:
            self.pose_cnn_results_test=None
            self.blender_to_bop_pose=None
        
    def load_random_background(self, im_observed, mask):
        VOC_root = os.path.join(self.VOC_ROOT, "VOCdevkit/VOC2012")
        VOC_image_set_dir = os.path.join(VOC_root, "ImageSets/Main")
        VOC_bg_list_path = os.path.join(VOC_image_set_dir, "diningtable_trainval.txt")
        with open(VOC_bg_list_path, "r") as f:
            VOC_bg_list = [
                line.strip("\r\n").split()[0] for line in f.readlines() if line.strip("\r\n").split()[1] == "1"
            ]
        height, width, channel = im_observed.shape
        target_size = min(height, width)
        max_size = max(height, width)
        observed_hw_ratio = float(height) / float(width)

        k = random.randint(0, len(VOC_bg_list) - 1)
        bg_idx = VOC_bg_list[k]
        bg_path = os.path.join(VOC_root, "JPEGImages/{}.jpg".format(bg_idx))
        bg_image = cv2.imread(bg_path, cv2.IMREAD_COLOR)[...,::-1] #RGB
        bg_h, bg_w, bg_c = bg_image.shape
        bg_image_resize = np.zeros((height, width, channel), dtype="uint8")
        if (float(height) / float(width) < 1 and float(bg_h) / float(bg_w) < 1) or (
            float(height) / float(width) >= 1 and float(bg_h) / float(bg_w) >= 1
        ):
            if bg_h >= bg_w:
                bg_h_new = int(np.ceil(bg_w * observed_hw_ratio))
                if bg_h_new < bg_h:
                    bg_image_crop = bg_image[0:bg_h_new, 0:bg_w, :]
                else:
                    bg_image_crop = bg_image
            else:
                bg_w_new = int(np.ceil(bg_h / observed_hw_ratio))
                if bg_w_new < bg_w:
                    bg_image_crop = bg_image[0:bg_h, 0:bg_w_new, :]
                else:
                    bg_image_crop = bg_image
        else:
            if bg_h >= bg_w:
                bg_h_new = int(np.ceil(bg_w * observed_hw_ratio))
                bg_image_crop = bg_image[0:bg_h_new, 0:bg_w, :]
            else:  # bg_h < bg_w
                bg_w_new = int(np.ceil(bg_h / observed_hw_ratio))
                print(bg_w_new)
                bg_image_crop = bg_image[0:bg_h, 0:bg_w_new, :]

        bg_image_resize_0, _ = resize(bg_image_crop, target_size, max_size)
        h, w, c = bg_image_resize_0.shape
        bg_image_resize[0:h, 0:w, :] = bg_image_resize_0

        # add background to image_observed
        res_image = bg_image_resize.copy()
        res_image[mask>0]=im_observed[mask>0]

        # im_observed = res_image
        return res_image

    def _read_data(self, idx):
        """
        info structure:
        {
            'cat':[
                {
                "index": idx,
                "model_path": str,
                "rgb_path": str,
                "depth_path": str,
                "RT": np.array([3,4]),
                "K":  np.array([3,3]),
                },
                {
                "index": idx,
                "model_path": str,
                "rgb_path": str,
                "depth_path": str,
                "RT": np.array([3,4]),
                "K":  np.array([3,3]),
                }
            ...
            ],
            'dog':[

            ]
            ...
        }

        """

        if isinstance(idx, (tuple, list)):
            idx, seed = idx
        else:
            seed = None

        seq_lengths = np.array(self.infos['seq_lengths'])
        seq_lengths_cum = np.cumsum(seq_lengths)
        seq_lengths_cum = np.insert(seq_lengths_cum, 0, 0)  # insert a dummy 0
        seq_idx = np.nonzero(seq_lengths_cum > idx)[0][0]-1

        frame_idx = idx - seq_lengths_cum[seq_idx]

        info = self.infos["data"][seq_idx]
        dataset_idx = self.infos["dataset_idx"][seq_idx]
        

        model_points_path = os.path.join(f'{os.path.dirname(__file__)}/../EXPDATA/LM6d_converted/models/{self.infos["seqs"][seq_idx]}/textured.obj' ) # TODO: need check

        rgb_path = os.path.join(self.root_paths[dataset_idx], info[frame_idx]['rgb_observed_path']) 
        depth_path = os.path.join(self.root_paths[dataset_idx], info[frame_idx]['depth_gt_observed_path']) 

        if info[frame_idx].get('rgb_noisy_rendered', None) is not None:
            rgb_noisy_rendered_path = os.path.join(self.root_paths[dataset_idx], info[frame_idx]['rgb_noisy_rendered']) 
        else:
            rgb_noisy_rendered_path = None
        if info[frame_idx].get('depth_noisy_rendered', None) is not None:
            depth_noisy_rendered_path = os.path.join(self.root_paths[dataset_idx], info[frame_idx]['depth_noisy_rendered']) 
        else:
            depth_noisy_rendered_path = None

        if info[frame_idx].get('pose_noisy_rendered', None) is not None:
            rendered_RT = info[frame_idx]['pose_noisy_rendered'].astype(np.float32)
        # else:
        elif self.is_train:
            rendered_RT = sample_poses( info[frame_idx]['gt_pose'].astype(np.float32)[None] )

        K = info[frame_idx]['K'].astype(np.float32)
        RT = info[frame_idx]['gt_pose'].astype(np.float32) #[R,t]

        # evaluation 
        if not self.is_train:
            if self.pose_cnn_results_test is not None:
                class_name=self.infos["seqs"][seq_idx]

                if self.init_pose_type == "PVNET_LINEMOD":
                    try:
                        posecnn_RT = self.pose_cnn_results_test[class_name][frame_idx] # if self.pose_cnn_results_test is not None else np.zeros_like(RT)
                        #Transformations are needed as the pvnet has a different coordinate system. 
                        posecnn_RT[:3,:3] =  posecnn_RT[:3,:3]@self.blender_to_bop_pose[class_name][:3,:3].T
                        posecnn_RT[:3,3:] =  -posecnn_RT[:3,:3] @self.blender_to_bop_pose[class_name][:3,3:]  + posecnn_RT[:3,3:] 
                    except:
                        print("Warning: frame_idx is out of the range of self.pose_cnn_results_test!", flush=True)
                        posecnn_RT= se3_q2m(self.pose_cnn_results_test_posecnn[class_name][frame_idx]['pose']) #np.zeros_like(RT)
                elif self.init_pose_type =="POSECNN_LINEMOD":
                    posecnn_RT= se3_q2m(self.pose_cnn_results_test_posecnn[class_name][frame_idx]['pose']) 
                elif self.init_pose_type == "PVNET_LINEMOD_OCC":
                    try:
                        posecnn_RT = self.pose_cnn_results_test[class_name][frame_idx].copy()# if self.pose_cnn_results_test is not None else np.zeros_like(RT)
                        #Transformations are needed as the pvnet has a different coordinate system. 
                        posecnn_RT[:3,:3] =  posecnn_RT[:3,:3]@self.blender_to_bop_pose[class_name][:3,:3].T
                        posecnn_RT[:3,3:] =  -posecnn_RT[:3,:3] @self.blender_to_bop_pose[class_name][:3,3:]  + posecnn_RT[:3,3:] 
                    except:
                        # print(frame_idx)
                        raise
                else:
                    raise NotImplementedError 
                
                rendered_RT = posecnn_RT
            else:
                print("Warning: fail to load cnn poses!", flush=True)
                posecnn_RT = np.zeros_like(RT)
        else:
            posecnn_RT = np.zeros_like(RT)
        
        #add noise--for testing purpose only, should always be disabled in normal cases 
#         rot_std=0; trans_std=0.04; ang_max=1000;
#         print(f"Add pose noises rot_std={rot_std}, trans_std={trans_std}", flush=True)
#         rendered_RT=sample_poses(rendered_RT[None], rot_std=rot_std, trans_std=trans_std, ang_max=ang_max) 

        # Regularize the matrix to be a valid rotation
        rendered_RT[:3,:3] = rendered_RT[:3,:3]@ np.linalg.inv(scipy.linalg.sqrtm(rendered_RT[:3,:3].T@rendered_RT[:3,:3]))
        
        # model_points = np.fromfile(
        #     str(model_points_path), dtype=np.float32, count=-1).reshape([-1, self.model_point_dim]) # N x model_point_dim
        model_points, _,_ = load_obj(str(model_points_path) )
        model_points = model_points.numpy()
        
        visb = model_points[:,-1:]  # N x model_point_dim

        model_point_features=np.ones_like(model_points[:,:1]).astype(np.float32)


        rgb =  np.asarray(PIL.Image.open(rgb_path))

        if depth_path.endswith('.npy'):
            depth = np.load(depth_path) # blender 
        else:
            depth = cv2.imread(depth_path, -1) /1000.

        if self.is_train and "LM6d_refine_syn" in self.root_paths[dataset_idx]: #synthetic data
            rgb = self.load_random_background(rgb, mask=(depth>0)[...,None].repeat(rgb.shape[-1], axis=-1) )


        
        rgb_rendered =  np.asarray(PIL.Image.open(rgb_noisy_rendered_path)) if rgb_noisy_rendered_path is not None else None
        depth_rendered = np.asarray(PIL.Image.open(depth_noisy_rendered_path))/1000 if depth_noisy_rendered_path is not None else None #TODO: need check

        ren_mask = render_pointcloud(model_points, rendered_RT[None],K=K[None], render_image_size=rgb.shape[:2] ).squeeze()>0
        # depth = range_to_depth(depth<1, depth*2, K)

        return {
            "class_name":  self.infos["seqs"][seq_idx], 
            "idx": idx,
            "model_points": model_points,
            "visibility": visb,
            "model_point_features":model_point_features,
            "image": rgb,
            "depth": depth,
            "mask": depth>0,
            "rendered_image": rgb_rendered,
            "rendered_depth": depth_rendered,
            "K": K,
            "RT": RT,
            "rendered_RT": rendered_RT.astype(np.float32),
            "ren_mask":ren_mask,
            "POSECNN_RT": posecnn_RT.astype(np.float32), #for test, TODO
            "scale": 1 # model_scale * scale = depth_scale
        }



    def __getitem__(self, idx):

        data=self._read_data(idx) 
        try:
            data_p=self.prep_func(data, rand_rgb_transformer=self.rgb_transformer, find_2d3d_correspondence=self.is_train )
        except Exception as e: 
            if e.args[0] in ["Too few correspondences are found!"] :
                if isinstance(idx, (tuple, list)):
                    # idx, seed = idx
                    idx = [(idx[0]+1)%self.__len__(), idx[1]]
                else:
                    idx = (idx+1) %self.__len__()
                data_p= self.__getitem__(idx )
            else:
                raise ValueError

        return data_p

    def __len__(self):
        return np.sum(self.infos['seq_lengths'])

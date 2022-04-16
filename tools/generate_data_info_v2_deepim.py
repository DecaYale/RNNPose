#The version compatible with deepim   
import os
import numpy as np
import copy
import pickle
import fire
import glob
import re


def parse_pose_file(file):
    with open(file) as f:
        lines = [line.strip() for line in f.readlines()]

    poses = []
    for line in lines:
        poses.append(
            np.array([np.float32(l) for l in line.split()],
                     dtype=np.float32).reshape((3, 4))
        )
    return poses


def parse_calib_file(file):
    info = {}
    with open(file) as f:
        lines = [line.strip() for line in f.readlines()]
    for i, l in enumerate(lines):
        nums = np.array([np.float32(x)
                         for x in l.split(' ')[1:]], dtype=np.float32)
        if i < 4:
            info[f"calib/P{i}"] = nums.reshape((3, 4))
        else:
            info[f"calib/Tr_velo_to_cam"] = nums.reshape((3, 4))
    return info


# def create_data_info(data_root, saving_path, is_test_data=False):
# def create_data_info(data_root, saving_path, data_type='train'):
def create_data_info(data_root, saving_path, training_data_ratio=0.8, shuffle=True, ):
    """[summary]
        info structure:
        {
            0:[
                {
                "index": idx,
                "lidar_bin_path": lidar_bin_paths[idx],
                "RT": poses[idx],
                "K": poses[idx],
                },
                {
                "index": idx,
                "lidar_bin_path": lidar_bin_paths[idx],
                "RT": poses[idx],
                "K": poses[idx],
               
                },
            }
            ...
            ],
            1:[

            ]
            ...
        }

    """

    image_dir = os.path.join(data_root, )
    pose_dir = os.path.join(data_root )
    depth_dir = os.path.join(data_root)
    # blender_to_bop_pose=np.load("/DATA/yxu/LINEMOD/metricpose/blender2bop_RT.npy", allow_pickle=True).flat[0]
    blender_to_bop_pose=np.load(f"{os.path.dirname(os.path.abspath(__file__)) }/../EXPDATA/init_poses/pose_conversion/blender2bop_RT.npy", allow_pickle=True).flat[0]
    # seqs=['cat']
    seqs=['cat', 'ape', 'cam', 'duck', 'glue', 'iron', 'phone','benchvise', 'can', 'driller', 'eggbox', 'holepuncher', 'lamp']
    print(seqs)
    max_items_per_seq=10000
    # create training data
    res = {}
    eval_res = {}
    for seq in seqs:
        res[seq] = []
        eval_res[seq]=[]

        image_path_dir = os.path.join(image_dir, seq)
        depth_path_dir = os.path.join(depth_dir, seq)
        pose_path_dir = os.path.join(pose_dir, seq)
        # image_paths = os.listdir(lidar_bin_dir)
        image_paths = glob.glob(r'{}/*.jpg'.format(image_path_dir) ) 
        depth_paths = glob.glob(r'{}/*depth*.npy'.format(depth_path_dir) ) 

        pose_paths = glob.glob(r'{}/*RT.pkl'.format(pose_path_dir) ) 
        #for compatibility
        if len(pose_paths) ==0:
            pose_paths = glob.glob(r'{}/*params.pkl'.format(pose_path_dir) ) 


        # image_paths.sort(key=lambda s: int(os.path.basename(s).split('.')[0]) )

        image_paths.sort(key=lambda s: int(re.split( '\.|_' ,os.path.basename(s))[0]) )
        depth_paths.sort(key=lambda s: int(os.path.basename(s).split('_')[0]))
        pose_paths.sort(key=lambda s: int(os.path.basename(s).split('_')[0]))


        data_num=len(image_paths[:max_items_per_seq] ) 
        if shuffle: 
            
            permute=np.random.permutation(data_num)
        else:
            permute = np.arange(data_num)
        train_split=permute[:int(data_num*training_data_ratio)]
        eval_split= permute[int(data_num*training_data_ratio):]

        # for idx in range(len(image_paths[:max_items_per_seq])):
        for idx in train_split:
            # print(image_paths[idx], depth_paths[idx])
            with open(pose_paths[idx],'rb') as f:
                pose = pickle.load(f) 

            # pose['K'] = np.array([[572.4114, 0., 325.2611],
            #                   [0., 573.57043, 242.04899],
            #                   [0., 0., 1.]])
            if seq=='cam':
                bl2bo = blender_to_bop_pose['camera']
            else:
                bl2bo = blender_to_bop_pose[seq]

            pose["RT"][:3,:3] =  pose["RT"][:3,:3]@bl2bo[:3,:3].T
            # pose["RT"][:3,3:] =  -pose["RT"][:3,:3]@bl2bo[:3,:3].T @bl2bo[:3,3:]  + pose["RT"][:3,3:] 
            pose["RT"][:3,3:] =  -pose["RT"][:3,:3] @bl2bo[:3,3:]  + pose["RT"][:3,3:] 
            info = {
                "index": idx,
                # "image_path": image_paths[idx].replace(image_dir+'/',''),
                # "depth_path": depth_paths[idx].replace(depth_dir+'/',''),
                "rgb_observed_path": image_paths[idx].replace(image_dir,'./'),
                "depth_gt_observed_path": depth_paths[idx].replace(depth_dir,'./'),
                "rgb_noisy_rendered": None,
                "depth_noisy_rendered": None,
                "pose_noisy_rendered": None,
                "model_points_path": f"{seq}.bin",
                # "RT": pose["RT"],
                "gt_pose":  pose["RT"],
                "K": pose["K"],
                "bbox":pose.get('bbox', None)
            }

            print(info['rgb_observed_path'], info['depth_gt_observed_path'], image_dir)#, bl2bo[:3,:3])
            res[seq].append(info)


    train_saving_path=saving_path+'.train'
    # eval_saving_path=saving_path+'.eval'
    with open(train_saving_path, 'wb+') as f:

        print("Total data amount:", np.sum([len(res[r]) for r in res]))
        pickle.dump(res, f)

    # with open(eval_saving_path, 'wb+') as f:

    #     print("Total data amount:", np.sum([len(eval_res[r]) for r in eval_res]))
    #     pickle.dump(eval_res, f)

if __name__ == '__main__':
    fire.Fire()

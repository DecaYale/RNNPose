import os
import numpy as np
import copy
import pickle
import fire
import glob
import re
from data.linemod import linemod_config


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
# def create_data_info(data_root, saving_path, training_data_ratio=0.8, shuffle=True, ):
def create_data_info(data_root, saving_path, with_assertion=True):
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
    idx2class = {
        1: "ape",
        2: "benchvise",
        # 3: 'bowl',
        4: "camera",
        5: "can",
        6: "cat",
        # 7: 'cup',
        8: "driller",
        9: "duck",
        10: "eggbox",
        11: "glue",
        12: "holepuncher",
        13: "iron",
        14: "lamp",
        15: "phone",
    }
    class2idx = dict([[idx2class[k],k ] for k in idx2class.keys() ])

    seqs=class2idx.keys()

    observed_dir = os.path.join('', 'data/observed')
    gt_observed_dir = os.path.join('', 'data/gt_observed')
    rendered_dir = os.path.join('', 'data/rendered')
    set_split_dir = os.path.join('','image_set/observed')

    # max_items_per_seq=10000#8000#100#10000#2000
    # create training data
    res = {}
   
    for seq in seqs:
        res[seq] = []

        rgb_orig_dir = os.path.join(observed_dir, f"{class2idx[seq]:02d}")        
        rgb_noisy_rendered_dir = os.path.join(rendered_dir, seq )        

        depth_orig_dir= os.path.join(observed_dir, f"{class2idx[seq]:02d}")    
        depth_rendered_dir= os.path.join(gt_observed_dir, seq)    
        depth_noisy_rendered_dir= os.path.join(rendered_dir, seq)    

        gt_pose_dir = os.path.join(gt_observed_dir, seq)
        noisy_pose_dir = os.path.join(rendered_dir, seq)

        label_dir = os.path.join(observed_dir,  f"{class2idx[seq]:02d}" )


        rgb_orig_paths = glob.glob(r'{}/*color.png'.format(rgb_orig_dir) ) 
        train_split_file=os.path.join(data_root,set_split_dir, f"{seq}_train.txt")
        
        with open(train_split_file, 'r') as f:
            train_split = f.readlines()
            train_split = [ int(t.split('/')[-1] ) for t in train_split]

        rgb_orig_paths.sort(key=lambda s: int(re.split( '\.|_|-' ,os.path.basename(s))[0]) )


        NUM_RENDERED=10
        
        for idx in train_split:
            
            #original data paths
            gt_pose=np.loadtxt(os.path.join(data_root,gt_pose_dir, f"{idx:06d}-pose.txt"), skiprows=1).reshape(3,4)
            rgb_orig = os.path.join(rgb_orig_dir, f"{idx:06d}-color.png" )
            depth_orig=os.path.join(depth_orig_dir, f"{idx:06d}-depth.png" )
            depth_rendered = os.path.join(depth_rendered_dir, f"{idx:06d}-depth.png")
            label_orig=os.path.join(label_dir, f"{idx:06d}-label.png" )

            #rendered data paths
            rgb_noisy_rendered = [os.path.join(rgb_noisy_rendered_dir, f"{idx:06d}_{i}-color.png" ) for i in range(NUM_RENDERED) ]
            depth_noisy_rendered = [os.path.join(depth_noisy_rendered_dir, f"{idx:06d}_{i}-depth.png" ) for i in range(NUM_RENDERED) ]

            pose_noisy_rendered = [os.path.join(data_root, noisy_pose_dir, f"{idx:06d}_{i}-pose.txt" ) for i in range(NUM_RENDERED) ]
            pose_noisy_rendered = [np.loadtxt(p, skiprows=1).reshape(3,4) for p in pose_noisy_rendered ]

            #generate data pairs

            for noisy_data_idx in range(NUM_RENDERED):
                if with_assertion:
                    assert os.path.exists(os.path.join(data_root, rgb_orig) ), os.path.join(data_root, rgb_orig) 
                    assert os.path.exists(os.path.join(data_root, depth_orig) ), os.path.join(data_root, depth_orig) 
                    assert os.path.exists(os.path.join(data_root, label_orig) ), os.path.join(data_root, label_orig) 
                    assert os.path.exists(os.path.join(data_root, rgb_noisy_rendered[noisy_data_idx]) ), os.path.join(data_root, rgb_noisy_rendered[noisy_data_idx]) 
                    assert os.path.exists(os.path.join(data_root, depth_noisy_rendered[noisy_data_idx] ) ), os.path.join(data_root, depth_noisy_rendered[noisy_data_idx] ) 

                info = {
                    "index": idx,
                    # "rgb_orig_path": rgb_orig,
                    "rgb_observed_path": rgb_orig,
                    "depth_observed_path": depth_orig,
                    "depth_gt_observed_path": depth_rendered,
                    "gt_pose": gt_pose,

                    "rgb_noisy_rendered": rgb_noisy_rendered[noisy_data_idx],
                    "depth_noisy_rendered": depth_noisy_rendered[noisy_data_idx],
                    "pose_noisy_rendered": pose_noisy_rendered[noisy_data_idx],

                    "model_points_path": f"{seq}.bin",
                    #legacy
                    "RT": gt_pose,
                    "K": linemod_config.linemod_K,
                }
                res[seq].append(info)

                print(info['rgb_observed_path'], info['rgb_noisy_rendered'])
    
    train_saving_path=saving_path+'.train'
    with open(train_saving_path, 'wb+') as f:
        print("Total data amount:", np.sum([len(res[r]) for r in res]))
        pickle.dump(res, f)

    # eval_saving_path=saving_path+'.eval'
    # with open(eval_saving_path, 'wb+') as f:
    #     print("Total data amount:", np.sum([len(test_res[r]) for r in test_res]))
    #     pickle.dump(test_res, f)


if __name__ == '__main__':
    fire.Fire()

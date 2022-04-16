import numpy as np 
import cv2
import pickle
import fire
import os
import argparse


linemod_K = np.array([[572.4114, 0., 325.2611],
                  [0., 573.57043, 242.04899],
                  [0., 0., 1.]])


blender_K = np.array([[700., 0., 320.],
                    [0., 700., 240.],
                    [0., 0., 1.]])


def range_to_depth(mask, range, K):
    '''
       Transform the range image to depth image
    '''
    f=K[0,0]
    cx=K[0,2]
    cy=K[1,2]

    ys_, xs_=np.nonzero(mask)
    rngs=range[ys_,xs_]
    xs,ys=np.asarray(xs_,np.float32)+0.5,np.asarray(ys_,np.float32)+0.5

    Zs=f*rngs/( f**2 + (cx-xs)**2 + (cy-ys)**2 )**0.5
    depth = np.zeros_like(range)
    depth[ys_,xs_] = Zs
    return  depth

def crop(image, depth, mask, K_old, margin_ratio=0.1, output_size=128 ):
    '''
        image: HxWx3
        mask: HxW
        K_old: 3x3
    '''

    H,W, _ = image.shape
    

    mask=mask.astype('uint8')*255
    _x,_y,_w,_h = cv2.boundingRect(mask) 

    center=[_x+_w/2, _y+_h/2]

    L=int (max(_w,_h)* (1+2*margin_ratio))


    x=max(0, int(center[0]- L/2) )
    y=max(0, int(center[1]- L/2) )
    
    crop=image[y:y+L, x:x+L]
    depth_crop=depth[y:y+L, x:x+L]

    w=h=L # actual crop size

    #automatically handle the "out of range" problem
    patch=np.zeros([h,w,3], dtype=image.dtype)
    depth_patch=np.ones([h,w], dtype=depth.dtype)
    try:
        xp = 0
        yp = 0
        patch[xp : xp+crop.shape[0], yp:yp+crop.shape[1] ] =  crop
        depth_patch[xp : xp+crop.shape[0], yp:yp+crop.shape[1] ] = depth_crop 
    except:
        import pdb 
        pdb.set_trace()
    patch=cv2.resize(patch, (output_size,output_size), interpolation=cv2.INTER_LINEAR )
    depth_patch=cv2.resize(depth_patch, (output_size,output_size), interpolation=cv2.INTER_NEAREST )

    #update the intrinsic parameters
    K_new=np.zeros_like(K_old)
    scale=output_size/L
    K_new[0,2] = (K_old[0,2]-x)*scale
    K_new[1,2] = (K_old[1,2]-y)*scale
    K_new[0,0] = K_old[0,0]*scale
    K_new[1,1] = K_old[1,1]*scale
    K_new[2,2] = 1

    return patch, depth_patch, K_new 

class DataFormatter(object):
    def __init__(self, data_type, data_info_path, crop_param=None ):
        assert data_type in ['LM_SYN_PVNET', "LM_SYN_PVNET_LMK",'LM_FUSE_PVNET','LM_FUSE_SINGLE_PVNET' ]
        self.data_type=data_type
        self.crop_param=crop_param
        with open(data_info_path, 'rb') as f:
            self.data_info=pickle.load(f)
        pass 

    
    def process(self, data_root,depth_root, save_root):

        if self.data_type == "LM_SYN_PVNET":
            self._proc_LM_SYN_PVNET(self.data_info, data_root, save_root)
        elif self.data_type=='LM_SYN_PVNET_LMK':
            self._proc_LM_SYN_PVNET_LMK(self.data_info, data_root, save_root)
        elif self.data_type=='LM_FUSE_PVNET':
            self._proc_LM_FUSE_PVNET(self.data_info, data_root,depth_root, save_root)
        elif self.data_type=='LM_FUSE_SINGLE_PVNET':
            self._proc_LM_FUSE_SINGLE_PVNET(self.data_info, data_root,depth_root, save_root)
        else:
            raise NotImplementedError

    def _proc_LM_SYN_PVNET(self, data_info, data_root, save_root):

        for seq in data_info:
            for idx in range(len(data_info[seq]) ):
                # info = {
                #     "index": idx,
                #     "image_path": image_paths[idx].replace(image_path_dir+'/',''),
                #     "depth_path": depth_paths[idx].replace(depth_path_dir+'/',''),
                #     "RT": pose["RT"],
                #     "K": pose["K"],
                # }
                info = data_info[seq][idx]
                # image=cv2.imread( os.path.join(data_root, seq, info['image_path']) )
                # depth=np.load(os.path.join(data_root, seq, info['depth_path'])) 
                image=cv2.imread( os.path.join(data_root,  info['image_path']) )
                depth=np.load(os.path.join(data_root,  info['depth_path'])) 
                # K_old = info["K"]
                K_old = blender_K.copy()

                # maximum depth value = 1, which indicates the invalid regions 

                hs,ws=np.nonzero(depth<1)
                hmin,hmax=np.min(hs),np.max(hs)
                wmin,wmax=np.min(ws),np.max(ws)
                bbox= [hmin, wmin, hmax, wmax]

                mask=depth<1
                #transform the range map to depth map 
                depth = range_to_depth(depth<1, depth*2, K_old)
                if self.crop_param is not None:
                    image, depth, K_new=crop(image, depth, mask, K_old, margin_ratio=self.crop_param['margin_ratio'], output_size=self.crop_param['output_size'] )
                else:
                    K_new = K_old


                print(info['image_path'], info['depth_path'])
                patch_save_path=os.path.join(save_root, seq, f"{info['index']:05d}.jpg")
                depth_patch_save_path=os.path.join(save_root, seq, f"{info['index']:05d}_depth.npy")
                pose_save_path=os.path.join(save_root, seq, f"{info['index']:05d}_params.pkl")

                if not os.path.exists(os.path.join(save_root, seq)):
                    os.makedirs(os.path.join(save_root, seq))
                #save
                cv2.imwrite(patch_save_path,image)
                np.save(depth_patch_save_path, depth)
                with open(pose_save_path, 'wb+') as f:
                    pickle.dump({
                        "RT": info["RT"],
                        "K": K_new,
                        "bbox": bbox
                    },f)

    def _proc_LM_SYN_PVNET_LMK(self, data_info, data_root, save_root):
        #cam intrinsic is LM 
    
        for seq in data_info:
            for idx in range(len(data_info[seq]) ):
                # info = {
                #     "index": idx,
                #     "image_path": image_paths[idx].replace(image_path_dir+'/',''),
                #     "depth_path": depth_paths[idx].replace(depth_path_dir+'/',''),
                #     "RT": pose["RT"],
                #     "K": pose["K"],
                # }
                info = data_info[seq][idx]

                image=cv2.imread( os.path.join(data_root,  info['image_path']) )
                depth=np.load(os.path.join(data_root,  info['depth_path'])) 
                with open( os.path.join(data_root,  info['image_path'].replace(".jpg", "_RT.pkl")), 'rb' ) as f:
                    old_params=pickle.load(f)
                # K_old = info["K"]
                K_old = old_params["K"] #linemod_K.copy()

                # maximum depth value = 1, which indicates the invalid regions 

                hs,ws=np.nonzero(depth<1)
                hmin,hmax=np.min(hs),np.max(hs)
                wmin,wmax=np.min(ws),np.max(ws)
                bbox= [hmin, wmin, hmax, wmax]

                mask=depth<1
                #transform the range map to depth map 
                depth = range_to_depth(depth<1, depth*2, K_old)
                if self.crop_param is not None:
                    image, depth, K_new=crop(image, depth, mask, K_old, margin_ratio=self.crop_param['margin_ratio'], output_size=self.crop_param['output_size'] )
                else:
                    K_new = K_old


                print(info['image_path'], info['depth_path'], "...")
                patch_save_path=os.path.join(save_root, seq, f"{info['index']:05d}.jpg")
                depth_patch_save_path=os.path.join(save_root, seq, f"{info['index']:05d}_depth.npy")
                pose_save_path=os.path.join(save_root, seq, f"{info['index']:05d}_params.pkl")

                if not os.path.exists(os.path.join(save_root, seq)):
                    os.makedirs(os.path.join(save_root, seq))
                #save
                cv2.imwrite(patch_save_path,image)
                np.save(depth_patch_save_path, depth)
                with open(pose_save_path, 'wb+') as f:
                    pickle.dump({
                        "RT": old_params["RT"] ,#info["RT"],
                        "K": K_new,
                        "bbox": bbox
                    },f)

    def _proc_LM_FUSE_PVNET(self, data_info, data_root, depth_root, save_root):
    
        # The class name list used during the fusing process, which is used to find the respective mask index  
        linemod_cls_names=['ape','cam','cat','duck','glue','iron','phone', 'benchvise','can','driller','eggbox','holepuncher','lamp']

        for seq in data_info:
            seq_idx = linemod_cls_names.index(seq)
            for idx in range(len(data_info[seq]) ):
                # info = {
                #     "index": idx,
                #     "image_path": image_paths[idx].replace(image_path_dir+'/',''),
                #     "depth_path": depth_paths[idx].replace(depth_path_dir+'/',''),
                #     "RT": pose["RT"],
                #     "K": pose["K"],
                # }
                info = data_info[seq][idx]
                # if info['image_path'] =='cat/2744.jpg':
                #     info =  data_info[seq][idx+1]

                with open(os.path.join(data_root,  info['image_path']).split('.jpg')[0].replace(seq,'')+'_info.pkl', 'rb'  ) as f:
                    fuse_info = pickle.load(f )

                image=cv2.imread( os.path.join(data_root,  info['image_path']).split('.jpg')[0].replace(seq,'')+'_rgb.jpg' )
                try: 
                    depth_idx = fuse_info[2][seq_idx]['img_idx']
                except:
                    import pdb 
                    pdb.set_trace()

                rendered_depth=np.load( os.path.dirname(os.path.join(depth_root, info['image_path']  ))+ f'/{depth_idx}_depth.png.npy'  ) 

                fuse_mask=cv2.imread( os.path.join(data_root,  info['image_path']).split('.jpg')[0].replace(seq,'')+'_mask.png', ) 
                fuse_mask = fuse_mask[...,0]==(seq_idx+1) # fuse mask id starts from 1

                # """
                #may have bug
                hs,ws=np.nonzero(rendered_depth<1)
                hmin,hmax=np.min(hs),np.max(hs)
                wmin,wmax=np.min(ws),np.max(ws)
                # """
                bbox= [hmin+fuse_info[0][seq_idx][0], wmin+fuse_info[0][seq_idx][1], hmax+fuse_info[0][seq_idx][0], wmax+fuse_info[0][seq_idx][1]]

                depth = np.ones_like(rendered_depth) 

                try:
                    depth[hmin+fuse_info[0][seq_idx][0]: fuse_info[0][seq_idx][0]+hmax+1, wmin+fuse_info[0][seq_idx][1]: wmax+fuse_info[0][seq_idx][1] +1] = rendered_depth[hmin:hmax+1, wmin:wmax+1]
                except:
                    print(info['image_path'],"failed!") 
                    continue
                    """
                    #TODO: temp fix, may fail
                    patch_save_path=os.path.join(save_root, seq, f"{info['index']:05d}_pat.jpg")
                    depth_patch_save_path=os.path.join(save_root, seq, f"{info['index']:05d}_pat_depth.npy")
                    pose_save_path=os.path.join(save_root, seq, f"{info['index']:05d}_RT.pkl")
                    cv2.imwrite(patch_save_path, patch)
                    np.save(depth_patch_save_path, depth_patch)
                    with open(pose_save_path, 'wb+') as f:
                        pickle.dump({
                            # "RT": fuse_info[1][seq_idx], #info["RT"],
                            "RT": fuse_info_old[1][seq_idx], #info["RT"],
                            "K": K_new
                        },f)
                    # import pdb 
                    # pdb.set_trace()
                    # instance_inds.append(len(instance_inds))
                    continue
                    """
                # fuse_info_old = copy.deepcopy(fuse_info)
                
                # K_old = info["K"]
                K_old = linemod_K.copy()
                K_old[0,2] = (K_old[0,2]+fuse_info[0][seq_idx][1])
                K_old[1,2] = (K_old[1,2]+fuse_info[0][seq_idx][0])

                # maximum depth value = 1, which indicates the invalid regions 
                mask=depth<1
                # transform the range map to depth map 
                depth = range_to_depth(mask, depth*2, K_old)

                # depth = depth* fuse_mask + (1-fuse_mask) # use 1's to indicate the invalid depths
                # depth = depth* fuse_mask  # use 0's to indicate the invalid depths
                depth = depth # keep all the depths including occluded ones
                

                if self.crop_param is not None:
                    # image, depth, K_new=crop(image, depth, mask, K_old, margin_ratio=0.1, output_size=128 )
                    image, depth, K_new=crop(image, depth, mask, K_old, margin_ratio=self.crop_param['margin_ratio'], output_size=self.crop_param['output_size'] )
                else:
                    K_new = K_old

                print(info['image_path'], info['depth_path'], bbox)
                patch_save_path=os.path.join(save_root, seq, f"{info['index']:05d}.jpg")
                depth_patch_save_path=os.path.join(save_root, seq, f"{info['index']:05d}_depth.npy")
                # pose_save_path=os.path.join(save_root, seq, f"{info['index']:05d}_RT.pkl")
                pose_save_path=os.path.join(save_root, seq, f"{info['index']:05d}_params.pkl")
                mask_visb_save_path = os.path.join(save_root, seq, f"{info['index']:05d}_mask_visb.png")

                if not os.path.exists(os.path.join(save_root, seq)):
                    os.makedirs(os.path.join(save_root, seq))
                #save
                cv2.imwrite(patch_save_path,image)
                cv2.imwrite(mask_visb_save_path, fuse_mask*255 )

                np.save(depth_patch_save_path, depth)
                with open(pose_save_path, 'wb+') as f:
                    pickle.dump({
                        "RT": fuse_info[1][seq_idx], #info["RT"],
                        "K": K_new,
                        "bbox": bbox
                    },f)
    
    def _proc_LM_FUSE_SINGLE_PVNET(self, data_info, data_root, depth_root, save_root):
        
        # The class name list used during the fusing process, which is used to find the respective mask index  
        linemod_cls_names=['ape','cam','cat','duck','glue','iron','phone', 'benchvise','can','driller','eggbox','holepuncher','lamp']

        for seq in data_info:
            # seq_idx = linemod_cls_names.index(seq)
            seq_idx = 0 
            for idx in range(len(data_info[seq]) ):
                # info = {
                #     "index": idx,
                #     "image_path": image_paths[idx].replace(image_path_dir+'/',''),
                #     "depth_path": depth_paths[idx].replace(depth_path_dir+'/',''),
                #     "RT": pose["RT"],
                #     "K": pose["K"],
                # }
                info = data_info[seq][idx]
                # if info['image_path'] =='cat/2744.jpg':
                #     info =  data_info[seq][idx+1]
                with open(os.path.join(data_root, info['image_path'].split('.jpg')[0]+'_info.pkl'  ), 'rb'  ) as f:
                    fuse_info = pickle.load(f )

                # image=cv2.imread( os.path.join(data_root,  seq, info['image_path'].split('.jpg')[0].replace(seq,'')+'_rgb.jpg' ) )
                image=cv2.imread( os.path.join(data_root,  info['image_path'].split('.jpg')[0]+'_rgb.jpg' ) )
                try: 
                    depth_idx = fuse_info[2][seq_idx]['img_idx']
                except:
                    import pdb 
                    pdb.set_trace()

                rendered_depth=np.load( os.path.dirname(os.path.join(depth_root, info['image_path']  ))+ f'/{depth_idx}_depth.png.npy'  ) 

                fuse_mask=cv2.imread( os.path.join(data_root,  info['image_path']).split('.jpg')[0]+'_mask.png', ) 
                fuse_mask = fuse_mask[...,0]==(seq_idx+1) # fuse mask id starts from 1

                # """
                #may have bug
                hs,ws=np.nonzero(rendered_depth<1)
                hmin,hmax=np.min(hs),np.max(hs)
                wmin,wmax=np.min(ws),np.max(ws)
                # """
                bbox= [hmin+fuse_info[0][seq_idx][0], wmin+fuse_info[0][seq_idx][1], hmax+fuse_info[0][seq_idx][0], wmax+fuse_info[0][seq_idx][1]]

                depth = np.ones_like(rendered_depth) 

                try:
                    depth[hmin+fuse_info[0][seq_idx][0]: fuse_info[0][seq_idx][0]+hmax+1, wmin+fuse_info[0][seq_idx][1]: wmax+fuse_info[0][seq_idx][1] +1] = rendered_depth[hmin:hmax+1, wmin:wmax+1]
                except:
                    import pdb 
                    pdb.set_trace()
                    print(info['image_path'],"failed!") 
                    continue
                    """
                    #TODO: temp fix, may fail
                    patch_save_path=os.path.join(save_root, seq, f"{info['index']:05d}_pat.jpg")
                    depth_patch_save_path=os.path.join(save_root, seq, f"{info['index']:05d}_pat_depth.npy")
                    pose_save_path=os.path.join(save_root, seq, f"{info['index']:05d}_RT.pkl")
                    cv2.imwrite(patch_save_path, patch)
                    np.save(depth_patch_save_path, depth_patch)
                    with open(pose_save_path, 'wb+') as f:
                        pickle.dump({
                            # "RT": fuse_info[1][seq_idx], #info["RT"],
                            "RT": fuse_info_old[1][seq_idx], #info["RT"],
                            "K": K_new
                        },f)
                    # import pdb 
                    # pdb.set_trace()
                    # instance_inds.append(len(instance_inds))
                    continue
                    """
                # fuse_info_old = copy.deepcopy(fuse_info)
                
                # K_old = info["K"]
                K_old = linemod_K.copy()
                K_old[0,2] = (K_old[0,2]+fuse_info[0][seq_idx][1])
                K_old[1,2] = (K_old[1,2]+fuse_info[0][seq_idx][0])

                # maximum depth value = 1, which indicates the invalid regions 
                mask=depth<1
                # transform the range map to depth map 
                depth = range_to_depth(mask, depth*2, K_old)

                # depth = depth* fuse_mask + (1-fuse_mask) # use 1's to indicate the invalid depths
                depth = depth* fuse_mask  # use 0's to indicate the invalid depths
                

                if self.crop_param is not None:
                    # image, depth, K_new=crop(image, depth, mask, K_old, margin_ratio=0.1, output_size=128 )
                    image, depth, K_new=crop(image, depth, mask, K_old, margin_ratio=self.crop_param['margin_ratio'], output_size=self.crop_param['output_size'] )
                else:
                    K_new = K_old

                print(info['image_path'], info['depth_path'], bbox)
                patch_save_path=os.path.join(save_root, seq, f"{info['index']:05d}.jpg")
                depth_patch_save_path=os.path.join(save_root, seq, f"{info['index']:05d}_depth.npy")
                # pose_save_path=os.path.join(save_root, seq, f"{info['index']:05d}_RT.pkl")
                pose_save_path=os.path.join(save_root, seq, f"{info['index']:05d}_params.pkl")

                if not os.path.exists(os.path.join(save_root, seq)):
                    os.makedirs(os.path.join(save_root, seq))
                #save
                cv2.imwrite(patch_save_path,image)
                np.save(depth_patch_save_path, depth)
                with open(pose_save_path, 'wb+') as f:
                    pickle.dump({
                        "RT": fuse_info[1][seq_idx], #info["RT"],
                        "K": K_new,
                        "bbox": bbox
                    },f)


def run(data_type,data_info_path, image_root, depth_root, save_dir, crop_param=None):
    df = DataFormatter(data_type,data_info_path)
    df.process(image_root, depth_root,save_dir)

if __name__=='__main__':
    fire.Fire(run)
    
















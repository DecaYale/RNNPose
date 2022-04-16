import numpy as np 
import cv2 
import copy

def vis_pointclouds_cv2(pc, K, win_size, init_transform=None, color=None, img=None):
    '''
    pc: input point cloud of shape Nx3
    K: camera intrinsic of shape 3x3
    win_size: visualization window size (Wx,Wy)
    '''
    x = (K@pc.T).T  #Nx3
    
    x = x/x[:,-1:]
    x = x.astype(np.int32)
    x[:,0] = np.where((x[:,0]<0) | (x[:,0]>=win_size[1]), np.zeros_like(x[:,0]), x[:,0])
    x[:,1] = np.where((x[:,1]<0) | (x[:,1]>=win_size[0]), np.zeros_like(x[:,1]), x[:,1])

    if img is None:
        img=np.zeros(list(win_size)+[3], dtype=np.uint8)
    if color is None:
        # color = [255, 255, 0]
        color = [255, 255, 0]

    img[x[:, 1], x[:, 0]] = color
    
    img[x[pc[:,-1]<0][:,1], x[pc[:,-1]<0][:,0] ] = [255,0,0]

    return img
   
def vis_2d_keypoints_cv2(img, keypoints, color=None):
    '''
    img: input point cloud of shape HxWx3
    keypoints: Nx2 , (x,y)
    '''

    keypoints = np.around(keypoints).astype(np.int32)
    img=copy.copy(img)
    if color is None:
        color = [255, 255, 0]
    img[keypoints[:,1], keypoints[:,0]] = color

    return img
   


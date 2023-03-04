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
   

def get_model_corners(model):
    min_x, max_x = np.min(model[:, 0]), np.max(model[:, 0])
    min_y, max_y = np.min(model[:, 1]), np.max(model[:, 1])
    min_z, max_z = np.min(model[:, 2]), np.max(model[:, 2])
    corners_3d = np.array([
        [min_x, min_y, min_z],
        [min_x, min_y, max_z],
        [min_x, max_y, min_z],
        [min_x, max_y, max_z],
        [max_x, min_y, min_z],
        [max_x, min_y, max_z],
        [max_x, max_y, min_z],
        [max_x, max_y, max_z],
    ])
    return corners_3d

def vis_pose_box(RT,K, model, background=None,fig=None, ax=None, title='', label='', x_label='x', y_label='y', color='g', dot='-'):
    
    if fig is None:
        if background is not None:
            # dpi = float(matplotlib.rcParams['figure.dpi'])
            dpi=100
            # print(float(matplotlib.rcParams['figure.dpi']))
            fig = plt.figure(figsize=[s/dpi for s in background.shape[:2]], dpi=dpi )
        else:
            fig = plt.figure()

    if ax is None:
        ax=fig.gca()
    # ax.set_axis_off()
    corner_3d=get_model_corners(model)
    corner_2d = project(corner_3d, K, RT)

    if background is not None:
        ax.imshow(background)
    ax.add_patch(patches.Polygon(
        xy=corner_2d[[0, 1, 3, 2, 0, 4, 6, 2]], fill=False, linewidth=1, edgecolor=color))
    ax.add_patch(patches.Polygon(
        xy=corner_2d[[5, 4, 6, 7, 5, 1, 3, 7]], fill=False, linewidth=1, edgecolor=color))  

    # line,=ax.plot(x, y, dot, color=color, linewidth=1)    
    # line.set_label(label)
    # ax.set_title(title)
    # ax.set_xlabel(x_label)
    # ax.set_ylabel(y_label)
    # ax.axis('equal')
    # if label !='':
    #     ax.legend()
    return fig, ax
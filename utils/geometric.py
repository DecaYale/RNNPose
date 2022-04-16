import numpy as np 


def range_to_depth(mask, range, K):
    '''
       Transform the range image to depth image
    '''
    f=K[0,0]
    cx=K[0,2]
    cy=K[1,2]

    ys_, xs_=np.nonzero(mask)
    rngs=range[ys_,xs_]
    # xs,ys=np.asarray(xs,np.float32),np.asarray(ys,np.float32)
    xs,ys=np.asarray(xs_,np.float32)+0.5,np.asarray(ys_,np.float32)+0.5

    Zs=f*rngs/( f**2 + (cx-xs)**2 + (cy-ys)**2 )**0.5
    depth = np.zeros_like(range)
    depth[ys_,xs_] = Zs
    return  depth

def mask_depth_to_point_cloud(mask,depth,K):
    '''
        lift the depth under the mask to 3D point clouds
    '''
    ys, xs=np.nonzero(mask)
    dpts=depth[ys,xs]
    # xs,ys=np.asarray(xs,np.float32),np.asarray(ys,np.float32)
    xs,ys=np.asarray(xs,np.float32)+0.5,np.asarray(ys,np.float32)+0.5
    xys=np.concatenate([xs[:,None],ys[:,None]],1)
    xys*=dpts[:,None]
    xyds=np.concatenate([xys,dpts[:,None]],1)
    pts=np.matmul(xyds,np.linalg.inv(K).transpose())
    return pts.astype(np.float32), np.stack([xs,ys], axis=-1 )

def chordal_distance(R1,R2):
    return np.sqrt(np.sum((R1-R2)*(R1-R2))) 

def rotation_angle(R1, R2):
    return 2*np.arcsin( chordal_distance(R1,R2)/np.sqrt(8) )

def render_pointcloud(pc, T, K, render_image_size):
        """
        Args:
            T: (B,3,4) or (B,4,4)
            K: (B,3,3)
            render_image_size (tuple): (h,w)
            near (float, optional):  Defaults to 0.1.
            far (int, optional): Defaults to 6.
            mode: 'bilinear' or 'neareast'
        """

        B=T.shape[0]

        # T = self.cam_opencv2pytch3d.to(device=T.device)@T

        ## X_cam = X_world R + t
        # R = T[...,:3,:3].transpose(-1,-2)
        R = T[...,:3,:3].transpose( [0,2,1] )
        t = T[...,:3,3]

        #render depths
        # vert_depths= (self.verts@R+t).squeeze(0)[...,2:]
        X_cam= (pc@R+t)#.squeeze(0)

        x=X_cam@K.transpose([0,2,1])  #BxNx3
        depth = x[...,-1]
        x = x/x[...,-1:]

        out = np.zeros([1,1, *render_image_size], dtype=R.dtype)
        out[:, :, 
            np.round(x[0, :, 1]).astype(np.int64).clip(0, out.shape[2]-1),
            np.round(x[0, :, 0]).astype(np.int64).clip(0, out.shape[3]-1)] = depth 

        return out #1x1xHxW
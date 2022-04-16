## Speed optimized: sharing the rasterization among different rendering process

import torch
import torch.nn as nn 
import torch.nn.functional as F 

import numpy 

from pytorch3d.renderer import (
    OpenGLPerspectiveCameras, look_at_view_transform, look_at_rotation,
    RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,
    camera_position_from_spherical_angles, HardPhongShader, PointLights,FoVPerspectiveCameras, PerspectiveCameras, SoftPhongShader, Materials 
) 
try:
    from pytorch3d.structures import Meshes, Textures
    use_textures = True
except:
    from pytorch3d.structures import Meshes
    from pytorch3d.renderer import TexturesVertex
    from pytorch3d.renderer import TexturesVertex as Textures

    use_textures = False

import pytorch3d.renderer.mesh.utils as utils
from pytorch3d.io import load_obj, load_ply, load_objs_as_meshes
from pytorch3d.renderer.mesh.rasterizer import Fragments

from plyfile import PlyData
from utils.furthest_point_sample import fragmentation_fps



def rasterize(R, T, meshes, rasterizer, blur_radius=0):
    # It will automatically update the camera settings -> R, T in rasterizer.camera
    fragments = rasterizer(meshes, R=R, T=T)

    # Copy from pytorch3D source code, try if it is necessary to do gradient decent
    if blur_radius > 0.0:
        clipped_bary_coords = utils._clip_barycentric_coordinates(
            fragments.bary_coords
        )
        clipped_zbuf = utils._interpolate_zbuf(
            fragments.pix_to_face, clipped_bary_coords, meshes
        )
        fragments = Fragments(
            bary_coords=clipped_bary_coords,
            zbuf=clipped_zbuf,
            dists=fragments.dists,
            pix_to_face=fragments.pix_to_face,
        )
    return fragments

def set_bary_coords_to_nearest(bary_coords_):
    ori_shape = bary_coords_.shape
    exr = bary_coords_ * (bary_coords_ < 0)
    bary_coords_ = bary_coords_.view(-1, bary_coords_.shape[-1])
    arg_max_idx = bary_coords_.argmax(1)
    return torch.zeros_like(bary_coords_).scatter(1, arg_max_idx.unsqueeze(1), 1.0).view(*ori_shape) + exr

class MeshRendererWithDepth(nn.Module):
    def __init__(self, rasterizer, shader):
        super().__init__()
        self.rasterizer = rasterizer
        self.shader = shader

    def to(self, device):
        # Rasterizer and shader have submodules which are not of type nn.Module
        self.rasterizer.to(device)
        self.shader.to(device)
        return self

    def forward(self, meshes_world, **kwargs) -> torch.Tensor:
        fragments = self.rasterizer(meshes_world, **kwargs)
        images = self.shader(fragments, meshes_world, **kwargs)
        return images, fragments.zbuf

class MeshRendererWithDepth_v2(nn.Module):
    def __init__(self, rasterizer, shader):
        super().__init__()
        self.rasterizer = rasterizer
        self.shader = shader

    # def to(self, device):
    def to(self, *args, **kwargs):
        if 'device' in kwargs.keys():
            device = kwargs['device']
        else:
            device = args[0]
        super().to(device)
        # Rasterizer and shader have submodules which are not of type nn.Module
        self.rasterizer.to(device)
        self.shader.to(device)
        return self

    def forward(self, meshes_world, **kwargs) -> torch.Tensor:
        if 'fragments' not in kwargs.keys() or kwargs['fragments'] is None: # sharing fragment results with others for speed, as the rasterizing process occupies most of time
            if 'fragments' in kwargs:
                del kwargs['fragments']
                
            fragments = self.rasterizer(meshes_world, **kwargs)
        else:
            fragments = kwargs['fragments']
            del kwargs['fragments']

        images = self.shader(fragments, meshes_world, **kwargs)
        return images, fragments.zbuf

class DiffRender(nn.Module):
    def __init__(self, mesh_path, render_texture=False):
        super().__init__()

        # self.mesh = mesh
        if mesh_path.endswith('.ply'):
            verts, faces = load_ply(mesh_path)
            self.mesh = Meshes(verts=[verts], faces=[faces])
        elif mesh_path.endswith('.obj'):
            verts, faces,_ = load_obj(mesh_path)
            # import pdb; pdb.set_trace()
            faces=faces.verts_idx
            self.mesh=load_objs_as_meshes([mesh_path])

        # self.mesh = Meshes(verts=verts, faces=faces, textures=None)
        self.verts = verts
        self.faces = faces
        # self.mesh = Meshes(verts=[verts], faces=[faces])
        # self.feature=feature
        self.cam_opencv2pytch3d = torch.tensor(
                                [[-1,0,0,0],
                                [0,-1,0, 0],
                                [0,0, 1, 0],
                                [0,0, 0, 1]], dtype=torch.float32
                                )
        self.render_texture = render_texture

        #get patch infos
        self.pat_centers, self.pat_center_inds,  self.vert_frag_ids= fragmentation_fps(verts.detach().cpu().numpy(), 64)
        self.pat_centers = torch.from_numpy(self.pat_centers)
        self.pat_center_inds = torch.from_numpy(self.pat_center_inds)
        self.vert_frag_ids = torch.from_numpy(self.vert_frag_ids)[...,None] #Nx1




    def to(self, *args, **kwargs):
        if 'device' in kwargs.keys():
            device = kwargs['device']
        else:
            device = args[0]
        super().to(device)
        # self.rasterizer.cameras = self.rasterizer.cameras.to(device)
        # self.face_memory = self.face_memory.to(device)
        self.mesh = self.mesh.to(device)
        self.verts = self.verts.to(device)
        self.faces = self.faces.to(device)
        self.pat_centers = self.pat_centers.to(device)
        self.pat_center_inds = self.pat_center_inds.to(device)
        self.vert_frag_ids = self.vert_frag_ids.to(device)

        
        # self.cam_opencv2pytch3d = self.cam_opencv2pytch3d.to(device=device)
        return self

    def get_patch_center_depths(self, T, K):

        #no need to pre-transform, as here we do not use pytorch3d rendering
        # T = self.cam_opencv2pytch3d.to(device=T.device)@T

        ## X_cam = X_world R + t
        R = T[...,:3,:3].transpose(-1,-2)
        t = T[...,:3,3]

        #render depths
        X_cam= (self.pat_centers@R+t) #BxKx3
        depth= X_cam[...,2:] #BxKx1
        x=X_cam@K.transpose(-1,-2)  #BxNx3
        x = x/x[...,-1:]
        img_coords= x[...,:2]


        return depth, img_coords 

    # Calculate interpolated maps -> [n, c, h, w]
    # face_memory.shape: [n_face, 3, c]
    @staticmethod
    def forward_interpolate(R, t, meshes, face_memory, rasterizer, blur_radius=0, mode='bilinear', return_depth=True):

        fragments = rasterize(R, t, meshes, rasterizer, blur_radius=blur_radius)

        # [n, h, w, 1, d]
        if mode == 'nearest':
            out_map = utils.interpolate_face_attributes(fragments.pix_to_face, set_bary_coords_to_nearest(fragments.bary_coords), face_memory)
        else:
            out_map = utils.interpolate_face_attributes(fragments.pix_to_face, fragments.bary_coords, face_memory)
        out_map = out_map.squeeze(dim=3)
        out_map = out_map.transpose(3, 2).transpose(2, 1)
        if return_depth:
            return out_map, fragments.zbuf.permute(0,3,1,2), fragments # depth
        else:
            return out_map, fragments

    def render_mesh(self,  T, K, render_image_size, near=0.1, far=6, lights=(1,1,-1), fragments=None ):
        B=T.shape[0]
        # face_attribute = vert_attribute[self.faces.long()]

        device = T.device
        T = self.cam_opencv2pytch3d.to(device=T.device)@T

        ## X_cam = X_world R + t
        R = T[...,:3,:3].transpose(-1,-2)
        t = T[...,:3,3]

        cameras = PerspectiveCameras(focal_length= torch.stack([K[:,0,0], K[:,1,1] ], dim=-1), 
            principal_point=K[:,:2,2],  R=R, T=t, image_size=[render_image_size]*B, in_ndc=False, device=device)
        lights = PointLights(device=device, location=[lights])

        raster_settings = RasterizationSettings(
            image_size=render_image_size,
            blur_radius=0.0,
            faces_per_pixel=1, #5,
            bin_size=None, #0
            perspective_correct=True
        )
        materials = Materials(
            device=device,
            # specular_color=[[0.0, 1.0, 0.0]],
            shininess=0
        )
        # renderer = MeshRendererWithDepth(
        renderer = MeshRendererWithDepth_v2(
            rasterizer=MeshRasterizer(
                cameras=cameras, 
                raster_settings=raster_settings
            ),
            shader=SoftPhongShader(
            # shader=SoftGouraudShader(
                device=device, 
                cameras=cameras,
                lights=lights, 
                blend_params=BlendParams(1e-4, 1e-4, (0, 0, 0))
            )
        )
        image,depth =renderer(self.mesh, lights=lights, materials=materials, fragments=fragments)

        return image.permute(0,3,1,2)[:,:3], depth.permute(0,3,1,2) # to BCHW

    def render_offset_map(self,  T, K, render_image_size, near=0.1, far=6):
        yy, xx = torch.meshgrid(torch.arange(render_image_size[0], device=T.device), torch.arange(render_image_size[1], device=T.device) )
        # xx = xx.to(dtype=torch.float32)
        # yy = yy.to(dtype=torch.float32)
        coords_grid = torch.stack( [ xx.to(dtype=torch.float32),  yy.to(dtype=torch.float32)], dim=-1 )

        #no need to pre-transform, as here we do not use pytorch3d rendering
        # T = self.cam_opencv2pytch3d.to(device=T.device)@T

        ## X_cam = X_world R + t
        R = T[...,:3,:3].transpose(-1,-2)
        t = T[...,:3,3]

        #render depths
        X_cam= (self.pat_centers@R+t)#.squeeze(0)[...,2:]
        x=X_cam@K.transpose(-1,-2)  #BxNx3
        x = x/x[...,-1:]

        offset = x[...,None,None,:2] - coords_grid #BxNx1x1x2-HxWx2
        
        return offset.permute(0,1,4,2,3) #BxNx2xHxW

    # def forward(self, face_attribute, T, K, render_image_size, near=0.1, far=6):
    def forward(self, vert_attribute, T, K, render_image_size, near=0.1, far=6, render_texture=None, mode='bilinear') :
        """
        Args:
            vert_attribute: (N,C)
            T: (B,3,4) or (B,4,4)
            K: (B,3,3)
            render_image_size (tuple): (h,w)
            near (float, optional):  Defaults to 0.1.
            far (int, optional): Defaults to 6.
        """

        # use default rendering settings 
        if render_texture is None:
            render_texture= self.render_texture 
            
        if vert_attribute is None:
            # only render the rgb image
            return self.render_mesh(T, K, render_image_size, near=0.1, far=6 )

        B=T.shape[0]
        face_attribute = vert_attribute[self.faces.long()]

        device = T.device

        T = self.cam_opencv2pytch3d.to(device=T.device)@T

        ## X_cam = X_world R + t
        R = T[...,:3,:3].transpose(-1,-2)
        t = T[...,:3,3]
        # t = -(R@T[...,:3,3:]).squeeze(-1)
        
        cameras = PerspectiveCameras(focal_length= torch.stack([K[:,0,0], K[:,1,1] ], dim=-1), 
            principal_point=K[:,:2,2], image_size=[render_image_size]*B, in_ndc=False, device=device)

        raster_settings = RasterizationSettings(
            image_size=render_image_size,
            blur_radius=0.0,
            faces_per_pixel=1,
            bin_size=None, #0
            perspective_correct=True
        )

        rasterizer = MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        )

        # forward_interpolate(R, T, meshes, face_memory, rasterizer, blur_radius=0, mode='bilinear')
        out_map, out_depth, fragments=self.forward_interpolate(R, t, self.mesh, face_attribute, rasterizer, blur_radius=0, mode=mode)
        
        if not render_texture:
            return out_map, out_depth
        else:
            ren_tex=self.render_mesh(T, K, render_image_size, near=0.1, far=6, fragments=fragments  )

            #The first 3 channels contain the rendered textures
            return torch.cat([ren_tex[0], out_map ], dim=1), out_depth

    def render_depth(self, T, K, render_image_size, near=0.1, far=6, mode='neareast'):
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
        device = T.device

        T = self.cam_opencv2pytch3d.to(device=T.device)@T

        ## X_cam = X_world R + t
        R = T[...,:3,:3].transpose(-1,-2)
        t = T[...,:3,3]
        cameras = PerspectiveCameras(focal_length= torch.stack([K[:,0,0], K[:,1,1] ], dim=-1), 
            principal_point=K[:,:2,2], image_size=[render_image_size]*B, in_ndc=False, device=device)

        raster_settings = RasterizationSettings(
            image_size=render_image_size,
            blur_radius=0.0,
            faces_per_pixel=1,
            bin_size=0
        )

        rasterizer = MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        )


        #render depths
        vert_depths= (self.verts@R+t).squeeze(0)[...,2:]
        face_depths = vert_depths[self.faces.long()]
        out_depth, _ =self.forward_interpolate(R, t, self.mesh, face_depths, rasterizer, blur_radius=0, mode='nearest', return_depth=False)

        return out_depth

    def render_pointcloud(self, T, K, render_image_size, near=0.1, far=6):
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
        device = T.device

        # T = self.cam_opencv2pytch3d.to(device=T.device)@T

        ## X_cam = X_world R + t
        R = T[...,:3,:3].transpose(-1,-2)
        t = T[...,:3,3]

        #render depths
        # vert_depths= (self.verts@R+t).squeeze(0)[...,2:]
        X_cam= (self.verts@R+t)#.squeeze(0)

        x=X_cam@K.transpose(-1,-2)  #BxNx3
        depth = x[...,-1]
        x = x/x[...,-1:]

        out = torch.zeros([1,1, *render_image_size], dtype=R.dtype, device=R.device)
        out[:, :, 
            torch.round(x[0, :, 1]).long().clamp(0, out.shape[2]-1),
            torch.round(x[0, :, 0]).long().clamp(0, out.shape[3]-1)] = depth 

        return out #1x1xHxW


class DiffRendererWrapper(nn.Module):
    def __init__(self, obj_paths, device="cuda", render_texture=False ):
        super().__init__()

        self.renderers = []
        for obj_path in obj_paths:
            self.renderers.append( 
                DiffRender(obj_path, render_texture).to(device=device)
            )

        self.renderers=nn.ModuleList(self.renderers)
        self.cls2idx=None #updated outside

    def get_patch_center_depths(self, model_names, T, K):
        
        depths= []
        image_coords= []
        for b,_ in enumerate(model_names):
            model_idx = self.cls2idx[model_names[b]]
            depth, img_coord = self.renderers[model_idx].get_patch_center_depths(T[b:b+1], K )
            depths.append(depth)
            image_coords.append(img_coord)
        
        return torch.cat(depths, dim=0), torch.cat(image_coords, dim=0)


    def render_offset_map(self, model_names,  T, K, render_image_size, near=0.1, far=6):
        offsets= []
        for b,_ in enumerate(model_names):
            model_idx = self.cls2idx[model_names[b]]

            offset = self.renderers[model_idx].render_offset_map(T[b:b+1], K[b:b+1], render_image_size, near, far )
            offsets.append(offset)
        
        return torch.cat(offsets, dim=0)

    def render_pat_id(self, model_names,  T, K, render_image_size, near=0.1, far=6):

        pat_ids= []
        for b,_ in enumerate(model_names):
            model_idx = self.cls2idx[model_names[b]]
            # face_pat_id = self.renderers[model_idx].vert_frag_ids[self.renderers[model_idx].faces.long()]
            
            pat_id,_ = self.renderers[model_idx].forward(self.renderers[model_idx].vert_frag_ids.float()+1,T[b:b+1], K[b:b+1], render_image_size, near, far, 'nearest' )
            pat_ids.append(pat_id-1) #+1 -1, set invalid parts as -1's  
        
        return torch.cat(pat_ids, dim=0)

    def render_depth(self, model_names,  T, K, render_image_size, near=0.1, far=6):
    
        depth_outputs= []
        for b,_ in enumerate(model_names):
            model_idx = self.cls2idx[model_names[b]]

            depth = self.renderers[model_idx].render_depth( T[b:b+1], K[b:b+1], render_image_size, near, far, 'nearest' )
            depth_outputs.append(depth)
        
        return torch.cat(depth_outputs, dim=0)
    def render_mesh(self, model_names,  T, K, render_image_size, near=0.1, far=6):

        outputs= []
        for b,_ in enumerate(model_names):
            model_idx = self.cls2idx[model_names[b]]

            img= self.renderers[model_idx].render_mesh( T[b:b+1], K[b:b+1], render_image_size, near, far, )[0]
            outputs.append(img)
        
        return torch.cat(outputs, dim=0)

    def render_pointcloud(self, model_names, T, K, render_image_size, near=0.1, far=6):
        outputs= []
        for b,_ in enumerate(model_names):
            model_idx = self.cls2idx[model_names[b]]
            depth = self.renderers[model_idx].render_pointcloud( T[b:b+1], K[b:b+1], render_image_size, near, far )
            outputs.append(depth)
        
        return torch.cat(outputs, dim=0)

    def forward(self, model_names,  vert_attribute, T, K, render_image_size, near=0.1, far=6, render_tex=False):

        map_outputs= []
        depth_outputs= []
        for b,_ in enumerate(model_names):
            model_idx = self.cls2idx[model_names[b]]

            feamap, depth= self.renderers[model_idx]( vert_attribute[b], T[b:b+1], K[b:b+1], render_image_size, near, far, render_texture=render_tex )

            map_outputs.append(feamap)
            depth_outputs.append(depth)
        return torch.cat(map_outputs, dim=0) , torch.cat(depth_outputs, dim=0)



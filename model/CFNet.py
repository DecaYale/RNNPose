

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from thirdparty.raft.update import BasicUpdateBlock
from thirdparty.raft.extractor import BasicEncoder
from thirdparty.raft.corr import CorrBlock, AlternateCorrBlock
from thirdparty.raft.utils.utils import bilinear_sampler, coords_grid, upflow

try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass


class ImageFeaEncoder(nn.Module):
    def __init__(self, input_dim=3, output_dim=256):
        super().__init__()
        self.fnet = BasicEncoder(output_dim=output_dim, norm_fn='instance', dropout=False, input_dim=input_dim)        

        if 1:#self.args.pretrained_model is not None:
            print("Loading the weights of RAFT...")
            import os             
            self.load_state_dict(
                #  torch.load(self.args.pretrained_model, map_location='cpu'), strict=False
                 torch.load( f"{os.path.dirname(os.path.abspath(__file__)) }/../weights/img_fea_enc.pth", map_location='cpu'), strict=True
            )
        else:
            print("ImageFeaEncoder will be trained from scratch...")

    def forward(self, image1, image2):
        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0

        image1 = image1.contiguous()
        image2 = image2.contiguous()
        with autocast(enabled=True):
            fmap1, fmap2 = self.fnet([image1, image2])
        return fmap1, fmap2


class GRU_CFUpdator(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        input_dim =  args.get("input_dim", 3)

        self.hidden_dim = hdim = 128
        self.context_dim = cdim = 128
        args.corr_levels = 4
        args.corr_radius = 4

        if 'alternate_corr' not in self.args:
            self.args.alternate_corr = False

        self.update_block = BasicUpdateBlock(self.args, hidden_dim=hdim)

        if self.args.pretrained_model is not None:
            print("Loading the weights of RAFT...")
            import os             
            self.load_state_dict(
                #  torch.load(self.args.pretrained_model, map_location='cpu'), strict=False
                 torch.load( f"{os.path.dirname(os.path.abspath(__file__)) }/../weights/gru_update.pth", map_location='cpu'), strict=True
            )
        else:
            print("GRU_CFUpdator will be trained from scratch...")


    
    
    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img, downsample_rate=8):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H//downsample_rate, W//downsample_rate).to(img.device)
        coords1 = coords_grid(N, H//downsample_rate, W//downsample_rate).to(img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def upsample_flow(self, flow, mask, upsample_scale=8):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, upsample_scale, upsample_scale, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(upsample_scale * flow, [3,3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, upsample_scale*H, upsample_scale*W)


    def forward(self, fmap1, fmap2, iters=1, flow_init=None, upsample=True, test_mode=False, context_fea=None, update_corr_fn=True):
        """ Estimate optical flow between pair of frames """

        hdim = self.hidden_dim
        cdim = self.context_dim

        if update_corr_fn: # need carful handling outside
            # run the feature network
            self.fmap1 = fmap1.float()
            self.fmap2 = fmap2.float()
            if self.args.alternate_corr:
                self.corr_fn = AlternateCorrBlock(self.fmap1, self.fmap2, radius=self.args.corr_radius)
            else:
                self.corr_fn = CorrBlock(self.fmap1, self.fmap2, radius=self.args.corr_radius)

        if update_corr_fn: 
            # run the context network
            with autocast(enabled=self.args.mixed_precision):
                assert context_fea is not None
                ds = context_fea.shape[-1]//self.fmap1.shape[-1]
                cnet = F.interpolate(context_fea, scale_factor=1/ds, mode='bilinear', align_corners=True)

                self.net, self.inp = torch.split(cnet, [hdim, cdim], dim=1)
                self.net = torch.tanh(self.net)
                self.inp = torch.relu(self.inp)

        # coords0, coords1 = self.initialize_flow(image1)
        coords0, coords1 = self.initialize_flow(flow_init)

        if flow_init is not None:
            ds = flow_init.shape[-1]//coords0.shape[-1]
            if ds !=1:
                flow_init /=ds
                flow_init = F.interpolate(flow_init, scale_factor=1/ds, mode='bilinear', align_corners=True)

            coords1 = coords1 + flow_init

        flow_predictions = []
        for itr in range(iters):
            coords1 = coords1.detach()
            corr = self.corr_fn(coords1) # index correlation volume

            flow = coords1 - coords0
            with autocast(enabled=self.args.mixed_precision):
                # net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)
                self.net, up_mask, delta_flow = self.update_block(self.net, self.inp, corr, flow)

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

            # upsample predictions
            if up_mask is None:
                flow_up = upflow(coords1 - coords0, scale=image1.shape[2]//coords0.shape[2],)
            else:
                if self.args.fea_net in ["bigdx4"]:
                    flow_up = self.upsample_flow(coords1 - coords0, up_mask, upsample_scale=4)
                else:
                    flow_up = self.upsample_flow(coords1 - coords0, up_mask)
            
            flow_predictions.append(flow_up)

        if test_mode:
            return coords1 - coords0, flow_up
            
        return flow_predictions


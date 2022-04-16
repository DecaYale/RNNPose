import torch 
import torch.nn as nn 


from kpconv.kpconv_blocks import *
import torch.nn.functional as F
import numpy as np
from kpconv.lib.utils import square_distance

class KPSuperpoint3Dv2(nn.Module):
    #remove useless channels
    def __init__(self, config):
        super().__init__()
        self.normalize_output=config.get('normalize_output', True)

        # build the architectures
        config.architecture = [
        'simple',
        'resnetb',
        ]
        for i in range(config.num_layers-1):
            config.architecture.append('resnetb_strided')
            config.architecture.append('resnetb')
            config.architecture.append('resnetb')
        for i in range(config.num_layers-2):
            config.architecture.append('nearest_upsample')
            config.architecture.append('unary')
        config.architecture.append('nearest_upsample')
        config.architecture.append('last_unary')

        ############
        # Parameters
        ############
        # Current radius of convolution and feature dimension
        layer = 0
        r = config.first_subsampling_dl * config.conv_radius
        in_dim = config.in_features_dim
        out_dim = config.first_feats_dim
        self.K = config.num_kernel_points
        self.epsilon = torch.nn.Parameter(torch.tensor(-5.0))
        self.final_feats_dim = config.final_feats_dim

        #####################
        # List Encoder blocks
        #####################
        # Save all block operations in a list of modules
        self.encoder_blocks = nn.ModuleList()
        self.encoder_skip_dims = []
        self.encoder_skips = []

        # Loop over consecutive blocks
        for block_i, block in enumerate(config.architecture):

            # Check equivariance
            if ('equivariant' in block) and (not out_dim % 3 == 0):
                raise ValueError('Equivariant block but features dimension is not a factor of 3')

            # Detect change to next layer for skip connection
            if np.any([tmp in block for tmp in ['pool', 'strided', 'upsample', 'global']]):
                self.encoder_skips.append(block_i)
                self.encoder_skip_dims.append(in_dim)

            # Detect upsampling block to stop
            if 'upsample' in block:
                break

            # Apply the good block function defining tf ops
            self.encoder_blocks.append(block_decider(block,
                                                    r,
                                                    in_dim,
                                                    out_dim,
                                                    layer,
                                                    config))

            # Update dimension of input from output
            if 'simple' in block:
                in_dim = out_dim // 2
            else:
                in_dim = out_dim

            # Detect change to a subsampled layer
            if 'pool' in block or 'strided' in block:
                # Update radius and feature dimension for next layer
                layer += 1
                r *= 2
                out_dim *= 2

        #####################
        # bottleneck layer 
        #####################
        botneck_feats_dim = config.gnn_feats_dim
        self.bottle = nn.Conv1d(in_dim, botneck_feats_dim,kernel_size=1,bias=True)
        # num_head = config.num_head
        self.proj_gnn = nn.Conv1d(botneck_feats_dim,botneck_feats_dim,kernel_size=1, bias=True)

        
        #####################
        # List Decoder blocks
        #####################
        out_dim = botneck_feats_dim # + 2

        # Save all block operations in a list of modules
        self.decoder_blocks = nn.ModuleList()
        self.decoder_concats = []

        # Find first upsampling block
        start_i = 0
        for block_i, block in enumerate(config.architecture):
            if 'upsample' in block:
                start_i = block_i
                break
        
        # Loop over consecutive blocks
        for block_i, block in enumerate(config.architecture[start_i:]):

            # Add dimension of skip connection concat
            if block_i > 0 and 'upsample' in config.architecture[start_i + block_i - 1]:
                in_dim += self.encoder_skip_dims[layer]
                self.decoder_concats.append(block_i)

            # Apply the good block function defining tf ops
            self.decoder_blocks.append(block_decider(block,
                                                    r,
                                                    in_dim,
                                                    out_dim,
                                                    layer,
                                                    config))

                

            # Update dimension of input from output
            in_dim = out_dim

            # Detect change to a subsampled layer
            if 'upsample' in block:
                # Update radius and feature dimension for next layer
                layer -= 1
                r *= 0.5
                out_dim = out_dim // 2
        return

    def regular_score(self,score):
        score = torch.where(torch.isnan(score), torch.zeros_like(score), score)
        score = torch.where(torch.isinf(score), torch.zeros_like(score), score)
        return score

    def forward_encoder(self, batch):
        # Get input features
        x = batch['features'].clone().detach()
        len_src_c = batch['stack_lengths'][-1][0]
        len_src_f = batch['stack_lengths'][0][0]
        pcd_c = batch['points'][-1]
        pcd_f = batch['points'][0]
        src_pcd_c, tgt_pcd_c = pcd_c[:len_src_c], pcd_c[len_src_c:]

        sigmoid = nn.Sigmoid()
        #################################
        # 1. encoder 
        skip_x = []
        for block_i, block_op in enumerate(self.encoder_blocks):
            if block_i in self.encoder_skips:
                skip_x.append(x)
            x = block_op(x, batch)

        #################################
        # 2. project the bottleneck features
        feats_c = x.transpose(0,1).unsqueeze(0)  #[1, C, N]
        feats_c = self.bottle(feats_c)  #[1, C, N]
        
        return feats_c,skip_x

    def forward_middle(self, x):

        feats_c = self.proj_gnn(x)   
        feats_gnn_raw = feats_c.squeeze(0).transpose(0,1)

        return feats_gnn_raw


    def forward_decoder(self, x, skip_x, batch ):
        sigmoid = nn.Sigmoid()
        for block_i, block_op in enumerate(self.decoder_blocks):
            if block_i in self.decoder_concats:
                x = torch.cat([x, skip_x.pop()], dim=1)
            x = block_op(x, batch)
            
        feats_f = x[:,:self.final_feats_dim]

        # normalise point-wise features
        if self.normalize_output:
            feats_f = F.normalize(feats_f, p=2, dim=1)

        return feats_f 

    def forward(self, batch):
        # Get input features
        feats_c,skip_x = self.forward_encoder(batch)
        x = self.forward_middle(feats_c)
        feats_f = self.forward_decoder(x, skip_x, batch)

        return feats_f





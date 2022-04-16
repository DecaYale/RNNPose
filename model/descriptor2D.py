from easydict import EasyDict as edict
from pathlib import Path
import torch
from torch import nn
from torchplus.nn.modules.common import Empty



class SuperPoint2D(nn.Module):
    """SuperPoint Convolutional Detector and Descriptor

    SuperPoint: Self-Supervised Interest Point Detection and
    Description. Daniel DeTone, Tomasz Malisiewicz, and Andrew
    Rabinovich. In CVPRW, 2019. https://arxiv.org/abs/1712.07629

    """
    default_config = {
        'descriptor_dim': 256,
        'nms_radius': 4,
        'keypoint_threshold': 0.005,
        'max_keypoints': -1,
        'remove_borders': 4,
        'saliency_score_normalization_fuc': 'sigmoid',
        "use_instance_norm": True
    }

    def __init__(self, config):
        super().__init__()
        self.default_config.update(config) 
        self.config=edict(self.default_config)

        self.normalize_output=config.get('normalize_output', True)

        self.saliency_score_normalization_fuc= self.config.saliency_score_normalization_fuc
        assert self.saliency_score_normalization_fuc in ['sigmoid', 'softmax']

        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        if self.config.use_instance_norm:
            self.Normalization=nn.InstanceNorm2d
        else:
            self.Normalization = Empty


        c1, c2, c3, c4, c5 = 64, 64, 128, 128, 256
        self.input_dim=config.input_dim
        # self.conv1a = nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
        self.conv1a = nn.Conv2d(config.input_dim, c1, kernel_size=3, stride=1, padding=1)
        self.conv1b = nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
        self.conv2a = nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
        self.conv2b = nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
        self.conv3a = nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
        self.conv3b = nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
        self.conv4a = nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
        self.conv4b = nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)

        self.convPa = nn.Sequential(
            nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1), 
            self.Normalization(c5)
        )

        self.convPb = nn.Conv2d(c5, 1, kernel_size=1, stride=1, padding=0)

        self.convDa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)

        self.convDb = nn.Conv2d(
            c5, self.config['descriptor_dim'],
            kernel_size=1, stride=1, padding=0)

        self.decode1=nn.Sequential(
            nn.Upsample(scale_factor=2,mode='bilinear'),
            nn.Conv2d(c4,c4, kernel_size=3, stride=1, padding=1),
            self.Normalization(c4),
            nn.ReLU())
        self.decode2=nn.Sequential(
            nn.Upsample(scale_factor=2,mode='bilinear'),
            nn.Conv2d(c4+c3,c4, kernel_size=3, stride=1, padding=1),
            self.Normalization(c4),
            nn.ReLU()
            )
        
        self.decode3=nn.Sequential(
            nn.Upsample(scale_factor=2,mode='bilinear'),
            nn.Conv2d(c4+c2,c4, kernel_size=3, stride=1, padding=1),
            self.Normalization(c4),
            nn.ReLU()
            )
        
        path = Path(__file__).parent.parent/ 'weights/superpoint_v1.pth'

        self.load_state_dict(torch.load(str(path), map_location='cpu' ), strict=False)

        mk = self.config['max_keypoints']
        if mk == 0 or mk < -1:
            raise ValueError('\"max_keypoints\" must be positive or \"-1\"')

        print('Loaded SuperPoint model')

    def load_state_dict(self,state_dict, strict=True):
 
        if not strict:
            updated_state_dict = {}
            model_dict = self.state_dict()
            for k, v in state_dict.items():
                if k in model_dict and v.shape == model_dict[k].shape:
                    updated_state_dict[k] = v
        else:
            updated_state_dict = state_dict
        super().load_state_dict(updated_state_dict, strict)


    def forward_encoder(self, x):
        if self.input_dim==1:
            x=x.mean(dim=1, keepdims=True) #
        x_skip=[]
        # Shared Encoder
        x = self.relu(self.conv1a(x))
        x = self.relu(self.conv1b(x))
        x_skip.append(x)
        x = self.pool(x)
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        x_skip.append(x)
        x = self.pool(x)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x_skip.append(x)
        x = self.pool(x)
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))

        return x, x_skip
    def forward_decoder(self, x, x_skip):
        #upsample first
        x = self.decode1(x)
        x=torch.cat([x, x_skip[-1]], dim=1)
        x = self.decode2(x)
        x=torch.cat([x, x_skip[-2]], dim=1)
        x = self.decode3(x)


        # Compute the dense keypoint scores
        cPa = self.relu(self.convPa(x))
        scores = self.convPb(cPa)

        if self.saliency_score_normalization_fuc == 'sigmoid':
            scores = nn.functional.sigmoid(scores)
        elif self.saliency_score_normalization_fuc == 'softmax':
            scores_shape = scores.shape
            scores = scores.reshape(*scores.shape[0:2], -1)
            scores = nn.functional.softmax(scores/1, dim=-1)
            scores= scores.reshape(*scores_shape)#.clone()
        else:
            raise ValueError

        keypoints=None

        # Compute the dense descriptors
        cDa = self.relu(self.convDa(x))
        descriptors = self.convDb(cDa)
        if self.normalize_output:
            descriptors = torch.nn.functional.normalize(descriptors, p=2, dim=1)

        return keypoints, scores, descriptors
    def forward(self, data):
        """ Compute keypoints, scores, descriptors for image """
        # Shared Encoder
        x, x_skip=self.forward_encoder(data)

        # Compute the dense keypoint scores
        keypoints, scores, descriptors = self.forward_decoder(x, x_skip)

        return {
            'keypoints': keypoints,
            'scores': scores,
            'descriptors': descriptors,
        }


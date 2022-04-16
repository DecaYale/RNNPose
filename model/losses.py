from sklearn.metrics import precision_recall_fscore_support
from thirdparty.kpconv.lib.utils import square_distance
from abc import ABCMeta, abstractmethod
import time

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torchplus
# from utils.pca import pca_tch

import kornia
# import self_voxelo.utils.pose_utils as pose_utils
import apex.amp as amp

# class Loss(object):


class Loss(nn.Module):
    """Abstract base class for loss functions."""
    __metaclass__ = ABCMeta

    def __init__(self, loss_weight=1):
        super(Loss, self).__init__()
        self._loss_weight = loss_weight

    # def __call__(self,
    def forward(self,
                prediction_tensor,
                target_tensor,
                ignore_nan_targets=False,
                scope=None,
                **params):
        """Call the loss function.

        Args:
          prediction_tensor: an N-d tensor of shape [batch, anchors, ...]
            representing predicted quantities.
          target_tensor: an N-d tensor of shape [batch, anchors, ...] representing
            regression or classification targets.
          ignore_nan_targets: whether to ignore nan targets in the loss computation.
            E.g. can be used if the target tensor is missing groundtruth data that
            shouldn't be factored into the loss.
          scope: Op scope name. Defaults to 'Loss' if None.
          **params: Additional keyword arguments for specific implementations of
                  the Loss.

        Returns:
          loss: a tensor representing the value of the loss function.
        """
        if ignore_nan_targets:
            target_tensor = torch.where(torch.isnan(target_tensor),
                                        prediction_tensor,
                                        target_tensor)
        # ret = self._compute_loss(prediction_tensor, target_tensor, **params)
        # if isinstance(ret, (list, tuple)):
        #     return [self._loss_weight*ret[0]] + list(ret[1:])
        # else:
        return self._loss_weight*self._compute_loss(prediction_tensor, target_tensor, **params)

    @abstractmethod
    @amp.float_function
    def _compute_loss(self, prediction_tensor, target_tensor, **params):
        """Method to be overridden by implementations.

        Args:
          prediction_tensor: a tensor representing predicted quantities
          target_tensor: a tensor representing regression or classification targets
          **params: Additional keyword arguments for specific implementations of
                  the Loss.

        Returns:
          loss: an N-d tensor of shape [batch, anchors, ...] containing the loss per
            anchor
        """

        raise NotImplementedError


class L2Loss(Loss):

    def __init__(self, loss_weight=1):
        super(L2Loss, self).__init__(loss_weight)

    def _compute_loss(self, prediction_tensor, target_tensor, mask=None):
        """Compute loss function.

        Args:
          prediction_tensor: A float tensor of shape [batch_size, num_anchors,
            code_size] representing the (encoded) predicted locations of objects.
          target_tensor: A float tensor of shape [batch_size, num_anchors,
            code_size] representing the regression targets

        Returns:
          loss: a float tensor of shape [batch_size, num_anchors] tensor
            representing the value of the loss function.
        """
        diff = prediction_tensor - target_tensor

        if mask is not None:
            mask = mask.expand_as(diff).byte()
            diff = diff[mask]
        # square_diff = 0.5 * weighted_diff * weighted_diff
        square_diff = diff * diff
        return square_diff.mean()


class AdaptiveWeightedL2Loss(Loss):

    def __init__(self, init_alpha, learn_alpha=True, loss_weight=1, focal_gamma=0):
        super(AdaptiveWeightedL2Loss, self).__init__(loss_weight)
        self.learn_alpha = learn_alpha
        self.alpha = nn.Parameter(torch.Tensor(
            [init_alpha]), requires_grad=learn_alpha)
        self.focal_gamma = focal_gamma
        # self.alpha_shift = -13  # -10# TODO: temporarily test

    def _compute_loss(self, prediction_tensor, target_tensor, mask=None, alpha=None, focal_gamma=None):
        """Compute loss function.

        Args:
          prediction_tensor: A float tensor of shape [batch_size, num_anchors,
            code_size] representing the (encoded) predicted locations of objects.
          target_tensor: A float tensor of shape [batch_size, num_anchors,
            code_size] representing the regression targets

        Returns:
          loss: a float tensor of shape [batch_size, num_anchors] tensor
            representing the value of the loss function.
        """

        if focal_gamma is None:
            focal_gamma = self.focal_gamma
        _alpha = self.alpha
        if mask is None:
            mask = torch.ones_like(target_tensor)
        else:
            mask = mask.expand_as(target_tensor)

        diff = prediction_tensor - target_tensor
        square_diff = (diff * diff) * mask

        # loss = square_diff.mean()
        input_shape = prediction_tensor.shape
        loss = torch.sum(square_diff, dim=list(range(1, len(input_shape)))) / \
            (torch.sum(mask, dim=list(range(1, len(input_shape)))) + 1e-12)  # (B,)

        focal_weight = (torch.exp(-_alpha) * loss)**focal_gamma
        focal_weight = focal_weight/(torch.sum(focal_weight) + 1e-12)

        loss = focal_weight*(torch.exp(-_alpha) * loss)
        loss = loss.sum() + _alpha
        return loss


class MetricLoss(nn.Module):
    """
    We evaluate both contrastive loss and circle loss
    """

    def __init__(self, configs, log_scale=16, pos_optimal=0.1, neg_optimal=1.4, ):
        super(MetricLoss, self).__init__()
        self.log_scale = log_scale
        self.pos_optimal = pos_optimal
        self.neg_optimal = neg_optimal

        self.pos_margin = configs.pos_margin
        self.neg_margin = configs.neg_margin
        self.max_points = configs.max_points

        self.safe_radius = configs.safe_radius
        self.matchability_radius = configs.matchability_radius
        # just to take care of the numeric precision
        self.pos_radius = configs.pos_radius + 0.001
        self.weight = configs.get('loss_weight', 1)

    def get_circle_loss(self, coords_dist, feats_dist):
        """
        Modified from: https://github.com/XuyangBai/D3Feat.pytorch
        """
        pos_mask = coords_dist < self.pos_radius
        neg_mask = coords_dist > self.safe_radius

        # get anchors that have both positive and negative pairs
        row_sel = ((pos_mask.sum(-1) > 0) * (neg_mask.sum(-1) > 0)).detach()
        col_sel = ((pos_mask.sum(-2) > 0) * (neg_mask.sum(-2) > 0)).detach()

        # get alpha for both positive and negative pairs
        pos_weight = feats_dist - 1e5 * \
            (~pos_mask).float()  # mask the non-positive
        # mask the uninformative positive
        pos_weight = (pos_weight - self.pos_optimal)
        pos_weight = torch.max(torch.zeros_like(
            pos_weight), pos_weight).detach()

        neg_weight = feats_dist + 1e5 * \
            (~neg_mask).float()  # mask the non-negative
        # mask the uninformative negative
        neg_weight = (self.neg_optimal - neg_weight)
        neg_weight = torch.max(torch.zeros_like(
            neg_weight), neg_weight).detach()

        lse_pos_row = torch.logsumexp(
            self.log_scale * (feats_dist - self.pos_margin) * pos_weight, dim=-1)
        lse_pos_col = torch.logsumexp(
            self.log_scale * (feats_dist - self.pos_margin) * pos_weight, dim=-2)

        lse_neg_row = torch.logsumexp(
            self.log_scale * (self.neg_margin - feats_dist) * neg_weight, dim=-1)
        lse_neg_col = torch.logsumexp(
            self.log_scale * (self.neg_margin - feats_dist) * neg_weight, dim=-2)

        loss_row = F.softplus(lse_pos_row + lse_neg_row)/self.log_scale
        loss_col = F.softplus(lse_pos_col + lse_neg_col)/self.log_scale

        circle_loss = (loss_row[row_sel].mean() + loss_col[col_sel].mean()) / 2

        return circle_loss

    def get_recall(self, coords_dist, feats_dist):
        """
        Get feature match recall, divided by number of true inliers
        """
        pos_mask = coords_dist < self.pos_radius
        n_gt_pos = (pos_mask.sum(-1) > 0).float().sum()+1e-12
        _, sel_idx = torch.min(feats_dist, -1)

        sel_dist = torch.gather(coords_dist, dim=-1,
                                index=sel_idx[:, None])[pos_mask.sum(-1) > 0]
        n_pred_pos = (sel_dist < self.pos_radius).float().sum()
        recall = n_pred_pos / n_gt_pos
        return recall

    def get_weighted_bce_loss(self, prediction, gt):
        loss = nn.BCELoss(reduction='none')

        class_loss = loss(prediction, gt)

        weights = torch.ones_like(gt)
        w_negative = gt.sum()/gt.size(0)
        w_positive = 1 - w_negative

        weights[gt >= 0.5] = w_positive
        weights[gt < 0.5] = w_negative
        w_class_loss = torch.mean(weights * class_loss)

        #######################################
        # get classification precision and recall
        predicted_labels = prediction.detach().cpu().round().numpy()
        cls_precision, cls_recall, _, _ = precision_recall_fscore_support(
            gt.cpu().numpy(), predicted_labels, average='binary')

        return w_class_loss, cls_precision, cls_recall

    def forward(self, src_pcd, tgt_pcd, src_feats, tgt_feats, correspondence, scores_overlap, scores_saliency, rot=None, trans=None):
        """
        Circle loss for metric learning, here we feed the positive pairs only
        Input:
            src_pcd:        [N, 3], pcd of the 3d model 
            tgt_pcd:        [M, 3], pcd of the lifted model from 2d depth
            rot:            [3, 3], rotation used to rotate the src_pcd to the current frame
            trans:          [3, 1], translation used to translate the src_pcd to the current frame
            src_feats:      [N, C]
            tgt_feats:      [M, C]
        """

        if rot is not None and trans is not None:
            src_pcd = (torch.matmul(rot, src_pcd.transpose(0, 1)) +
                       trans).transpose(0, 1)

        stats = dict()

        #######################################
        # filter some of correspondence
        if(correspondence.size(0) > self.max_points):
            choice = np.random.permutation(correspondence.size(0))[
                :self.max_points]
            correspondence = correspondence[choice]

        src_idx = correspondence[:, 0]
        tgt_idx = correspondence[:, 1]
        src_pcd, tgt_pcd = src_pcd[src_idx], tgt_pcd[tgt_idx]
        src_feats, tgt_feats = src_feats[src_idx], tgt_feats[tgt_idx]

        #######################
        # get L2 distance between source / target point cloud

        coords_dist = torch.sqrt(square_distance(
            src_pcd[None, :, :], tgt_pcd[None, :, :]).squeeze(0))
        feats_dist = torch.sqrt(square_distance(
            src_feats[None, :, :], tgt_feats[None, :, :], normalised=True)).squeeze(0)

        ##############################
        # get FMR and circle loss
        ##############################
        recall = self.get_recall(coords_dist, feats_dist)
        circle_loss = self.get_circle_loss(coords_dist, feats_dist)

        stats['circle_loss'] = circle_loss
        stats['recall'] = recall

        return stats


class PointAlignmentLoss(nn.Module):
    def __init__(self, loss_weight=1, ):
        super().__init__()
        self._loss_weight = loss_weight

    def forward(self, R_pred, t_pred, R_tgt, t_tgt, points):
        return self._loss_weight*self._compute_loss(R_pred, t_pred, R_tgt, t_tgt, points)

    def _compute_loss(self, R_pred, t_pred, R_tgt, t_tgt, points, ):
        """[summary]

        Args:
            R_pred ([type]): [Bx3x3]
            t_pred ([type]): [Bx3]
            R_tgt ([type]): [Bx3x3]
            t_tgt ([type]): [Bx3]
            points ([type]): [BxNx3]

        Returns:
            loss [type]: [loss value]
        """

        loss = 0
        for b in range(len(points)):

            diff = points[b]@R_pred[b].transpose(-1, -2) + t_pred[b] - (
                points[b]@R_tgt[b].transpose(-1, -2)+t_tgt[b])

            square_diff = diff.abs()  
            loss += torch.mean(square_diff)*3

        # loss/=len(points)

        return loss

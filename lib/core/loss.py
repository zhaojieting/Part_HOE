from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Function
import numpy as np



class JointsOffsetLoss(nn.Module):
    def __init__(self, use_target_weight, offset_weight, smooth_l1):
        super(JointsOffsetLoss, self).__init__()
        self.use_target_weight = use_target_weight
        self.offset_weight = offset_weight
        self.criterion = nn.MSELoss(reduction='mean')
        self.criterion_offset = nn.SmoothL1Loss(reduction='mean') if smooth_l1 else nn.L1Loss(reduction='mean')

    def forward(self, heatmaps, offsetmaps, target, target_offset, target_weight):
        """
        calculate loss
        :param output: [batch, joints, height, width]
        :param hm_hps: [batch, 2*joints, height, width]
        :param target: [batch, joints, height, width]
        :param target_offset: [batch, 2*joints, height, width]
        :param mask_01: [batch, joints, height, width]
        :param mask_g: [batch, joints, height, width]
        :param target_weight: [batch, joints, 1]
        :return: loss=joint_loss+weight*offset_loss
        """
        batch_size, num_joints, _, _ = heatmaps.shape

        heatmaps_pred = heatmaps.reshape((batch_size, num_joints, -1)).split(1, dim=1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, dim=1)
        offsets_pred = offsetmaps.reshape((batch_size, 2*num_joints, -1)).split(2, dim=1)
        offsets_gt = target_offset.reshape((batch_size, 2*num_joints, -1)).split(2, dim=1)

        del batch_size, _

        joint_l2_loss, offset_loss = 0.0, 0.0

        for idx in range(num_joints):
            offset_pred = offsets_pred[idx] * heatmaps_gt[idx]  # [batch_size, 2, h*w]
            offset_gt = offsets_gt[idx] * heatmaps_gt[idx]      # [batch_size, 2, h*w]
            heatmap_pred = heatmaps_pred[idx].squeeze()     # [batch_size, h*w]
            heatmap_gt = heatmaps_gt[idx].squeeze()         # [batch_size, h*w]

            if self.use_target_weight:
                joint_l2_loss += 0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                )
                offset_loss += self.criterion_offset(
                    offset_pred.mul(target_weight[:, idx, None]),
                    offset_gt.mul(target_weight[:, idx, None])
                )  # target_weight[:, idx].unsqueeze(2)
            else:
                joint_l2_loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)
                offset_loss += self.criterion_offset(offset_pred, offset_gt)

        loss = joint_l2_loss + self.offset_weight * offset_loss

        return loss / num_joints, offset_loss / num_joints

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=70, contrast_mode='all',
                 base_temperature=70):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))
        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

class DistanceWeightedTripletLoss(torch.nn.Module):
    """
    Distance-weighted triplet loss for regression tasks.
    """

    def __init__(self, criterion, margin=1.0):
        super(DistanceWeightedTripletLoss, self).__init__()
        self.mse = criterion
        self.margin = margin

    def forward(self, anchor, positive, negative, anchor_regression, positive_regression, negative_regression):
        # Compute distances between anchor and positive/negative samples
        pos_dist = self.mse(anchor, positive)
        neg_dist = self.mse(anchor, negative)

        # Compute differences between predicted and actual regression values
        pos_reg_diff = self.mse(positive_regression, anchor_regression)
        neg_reg_diff = self.mse(negative_regression, anchor_regression)

        # Compute loss weights based on regression value differences
        weight = torch.exp(-torch.max(pos_reg_diff - neg_reg_diff + self.margin, torch.zeros_like(pos_reg_diff)))

        # Compute triple loss with weighted samples
        loss_triplet = weight * (pos_dist - neg_dist + self.margin)
        loss_triplet = torch.mean(torch.clamp(loss_triplet, min=0.0))

        return loss_triplet


# class CrossEntropy(nn.Module):
#     def __init__(self, ignore_label=-1, weight=None):
#         super(CrossEntropy, self).__init__()
#         self.ignore_label = ignore_label
#         self.criterion = nn.MSELoss(weight=weight,
#                                              ignore_index=ignore_label)

#     def forward(self, score, target):
#         ph, pw = score.size(2), score.size(3)
#         h, w = target.size(1), target.size(2)
#         if ph != h or pw != w:
#             score = F.upsample(input=score, size=(h, w), mode='bilinear', align_corners=True)

#         loss = self.criterion(score, target)

#         return loss


class BodyPartAttentionLoss(nn.Module):
    """ A body part attention loss as described in our paper
    'Somers V. & al, Body Part-Based Representation Learning for Occluded Person Re-Identification, WACV23'.
    Source: https://github.com/VlSomers/bpbreid
    """
    def __init__(self, loss_type='cl', label_smoothing=0.1, weight=None):
        super().__init__()
        self.weight = weight
        self.part_prediction_loss_1 = nn.CrossEntropyLoss(label_smoothing=label_smoothing, weight=self.weight)

    def forward(self, pixels_cls_scores, targets, code):
        """ Compute loss for body part attention prediction.
            Args:
                pixels_cls_scores [N, K, H, W]
                targets [N, H, W]
            Returns:
        """
        pixels_cls_loss = self.compute_pixels_cls_loss(pixels_cls_scores, targets, code)

        return pixels_cls_loss

    def compute_pixels_cls_loss(self, pixels_cls_scores, targets, code):
        pixels_cls_scores = pixels_cls_scores[code!=0]
        targets = targets[code!=0]
        if pixels_cls_scores.is_cuda:
            targets = targets.cuda()
        pixels_cls_score_targets = targets.flatten()  # [N*Hf*Wf]
        pixels_cls_scores = pixels_cls_scores.permute(0, 2, 3, 1).flatten(0, 2)  # [N*Hf*Wf, M]
        loss = self.part_prediction_loss_1(pixels_cls_scores, pixels_cls_score_targets.type(torch.LongTensor).cuda())
        return 1e-4*loss

class HwMasksCrossEntropy(nn.Module):
    def __init__(self, weight=None):
        super(HwMasksCrossEntropy, self).__init__()
        self.criterion = nn.CrossEntropyLoss(weight=weight,
                                             reduction='none')

    def forward(self, h_fea, w_fea, mask_gt, mask_code):
        loss = 0.0
        batch_size, num_part, height, width = mask_gt.shape

        mask_gt = mask_gt[mask_code!=0]
        hgt = torch.sum(mask_gt, dim=3)
        wgt = torch.sum(mask_gt, dim=2)

        n,c,h = hgt.size()
        w = wgt.size()[2]
        h_fea = h_fea[mask_code!=0]
        scale_hpred = h_fea.unsqueeze(3)            #n,c,h,1
        scale_hpred = F.interpolate(input=scale_hpred, size=(h,1),mode='bilinear', align_corners=True)
        scale_hpred = scale_hpred.squeeze(3)        #n,c,h
        # hgt = hgt[:,1:,:]
        # scale_hpred=scale_hpred[:,1:,:]
        hloss = torch.mean( ( hgt - scale_hpred ) * ( hgt - scale_hpred ) )

        w_fea = w_fea[mask_code!=0]
        scale_wpred = w_fea.unsqueeze(2)            #n,c,1,w
        scale_wpred = F.interpolate(input=scale_wpred, size=(1,w),mode='bilinear', align_corners=True)
        scale_wpred = scale_wpred.squeeze(2)        #n,c,w    
        # wgt=wgt[:,1:,:]   
        # scale_wpred = scale_wpred[:,1:,:]
        wloss = torch.mean( ( wgt - scale_wpred ) * ( wgt - scale_wpred ) ) 
        loss =  hloss + wloss 
        return 0.000001 * loss


class MasksCrossEntropy(nn.Module):
    def __init__(self, ignore_label=-1, weight=None):
        super(MasksCrossEntropy, self).__init__()
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(weight=weight,
                                            # label_smoothing=0.05,
                                             reduction='none')
        # self.criterion = nn.NLLLoss(weight=weight, reduction='none')

    def forward(self, pred_masks, targets, mask_code):
        loss = 0.0
        ph, pw = pred_masks.size(2), pred_masks.size(3)
        h, w = targets.size(1), targets.size(2)
        if ph != h or pw != w:
            pred_masks = F.upsample(input=pred_masks, size=(h, w), mode='bilinear', align_corners=True)
        # batch_size, num_parts, height, width = targets.shape
        batch_size, height, width = targets.shape

        # # 单个mask做cross entropy, 自己和背景分
        # targets = targets.reshape((batch_size, num_parts, -1)).split(1, 1)
        # pred_masks = pred_masks.reshape((batch_size, num_parts, -1)).split(1, 1)

        # for idx in range(num_parts):
        #     pred_mask = pred_masks[idx].squeeze()
        #     target = targets[idx].squeeze()
        #     loss_tmp = 1/(height*width) * self.criterion(pred_mask, target)
        #     loss += loss_tmp[mask_code == 1].mean()

        # 不同部分之间分类
        loss = self.criterion(pred_masks, targets.type(torch.LongTensor).cuda())[mask_code==1].mean()

        return 0.001 * loss

class VhConstraintLoss(nn.Module):
    def __init__(self):
        super(VhConstraintLoss, self).__init__()
        self.criterion = torch.nn.MSELoss(reduction='none')

    def forward(self, vertical_output, horizontal_output):
        horizontal_output = horizontal_output.detach()
        vertical = (vertical_output* torch.arange(37).cuda()).sum(keepdims=True, axis=1)
        horizontal = (horizontal_output* torch.arange(37).cuda()).sum(keepdims=True, axis=1)
        vertical = torch.abs(18 - vertical)
        horizontal = torch.abs(18 - horizontal)
        data = torch.abs(18 -(vertical+horizontal))
        # print(data)
        loss = data.mean()

        return 0.0001*loss


class HoeMSELoss(nn.Module):
    def __init__(self):
        super(HoeMSELoss, self).__init__()
        self.criterion = torch.nn.MSELoss(reduction='none')

    def forward(self, hoe_output, gt_degree, conf=None, weight=None):
        batch_size = hoe_output.size(0)
        loss = self.criterion(hoe_output,gt_degree)
        if weight != None:
            loss = loss[weight[:,0]!=0]
        if conf != None:
            # import pdb;pdb.set_trace()
            loss = loss * conf.expand_as(loss)
        return loss.mean()

class MyHoeMSELoss(nn.Module):
    def __init__(self):
        super(HoeMSELoss, self).__init__()
        self.criterion = torch.nn.MSELoss(reduction='none')

    def forward(self, hoe_output, gt_degree, weight=None):
        batch_size = hoe_output.size(0)
        loss = self.criterion(hoe_output,gt_degree)
        if weight != None:
            loss = loss[weight[:,0]!=0]
        return loss.mean()

class JointsMSELoss(nn.Module):
    def __init__(self, use_target_weight):
        super(JointsMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='mean')
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        loss = 0

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                loss += self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                )
            else:
                loss += self.criterion(heatmap_pred, heatmap_gt)

        return loss / num_joints

class SpatialConsistencyLoss(nn.Module):
    def __init__(self):
        super(SpatialConsistencyLoss, self).__init__()
        self.num_parts = 8
        self.num_joints = 23
        self.criterion = nn.MSELoss(reduction='none')

        # # Define the mapping of joints to parts
        # self.joints_to_parts = {
        #     0: [0,1,2,3,4],  # Assign joints to "head" part
        #     1: [5,6,7,8,9,10,11,12],  # Assign joints to "torso" part
        #     2: [11,12,13,14,15,16],  # Assign joints to "leg" part
        #     3: [15,16,17,18,19,20,21,22],  # Assign joints to "feet" part
        # }

        self.joints_to_parts = {
            0: [0,1,2,3,4],  # Assign joints to "head" part
            1: [5,6,11,12],  # Assign joints to "torso" part
            2: [5,7,9],  # Assign joints to "left arm" part
            3: [6,8,10],  # Assign joints to "right arm" part
            4: [11,13,15],  # Assign joints to "left leg" part
            5: [12,14,16],  # Assign joints to "right leg" part
            6: [15,17,18,19],  # Assign joints to "left feet" part
            7: [16,20,21,22],  # Assign joints to "right feet" part
        }


    def forward(self, pred_masks, pred_keypoints):
        loss = 0.0
        pred_keypoints.requires_grad = False
        mask_centers = self.calculate_part_centers(pred_masks)
        mask_centers = mask_centers.transpose(1,0)
        keypoint_centers = self.calculate_keypoint_centers(pred_keypoints)
        keypoint_centers = keypoint_centers.transpose(1,0)
        code = torch.any((keypoint_centers == 0) | (mask_centers == 0), dim=2)
        loss = self.criterion(mask_centers,keypoint_centers)
        return 1e-5*loss[code==False].mean()

    def calculate_part_centers(self, masks):
        b, c, h, w = masks.size()

        centers = torch.zeros((self.num_parts, b, 2), device=masks.device)
        
        # Find the coordinates of the center of each part
        for i in range(self.num_parts):
            mask = masks[:,i+1]
            centers[i] = self.calculate_center(mask, 0.5)
        return centers

    def calculate_keypoint_centers(self, tensor):
        b, k, h, w = tensor.size()

        centers = torch.zeros((self.num_parts, b, 2), device=tensor.device)
        for j in range(self.num_parts):
            part_heatmap = torch.sum(tensor[:,self.joints_to_parts[j],:,:], dim=1)
            centers[j] = self.calculate_center(part_heatmap, 0.3)

        return centers

    def calculate_center(self, mask, thresh_hold):
        batch_size = mask.size(0)
        center = torch.zeros((batch_size, 2), device=mask.device)
        # Find indices where mask value > 0.5
        indices = torch.nonzero(mask > thresh_hold, as_tuple=False)
        # Calculate center coordinates for the mask
        for i in range(batch_size):
            center_x = indices[indices[:, 0] == i][:,1].float().mean().item()
            center_y = indices[indices[:, 0] == i][:,2].float().mean().item()
            if center_x > 0:
                center[i,0] = center_x
            if center_y > 0:
                center[i,1] = center_y
        return center

class MaskIOU_loss(nn.Module):
    def __init__(self):
        super(MaskIOU_loss, self).__init__()
        self.criterion = nn.MSELoss(reduction='none')

    def forward(self, predict_mask):
        zero_tensor = torch.tensor(0.0).cuda()
        zero_tensor.requires_grad = True
        one_tensor = torch.tensor(1.0).cuda()
        one_tensor.requires_grad = True

        # left_mask = torch.where(predict_mask[:, 1, :, :] > 0.5, one_tensor, zero_tensor)
        # right_mask = torch.where(predict_mask[:, 2, :, :] > 0.5, one_tensor, zero_tensor)
        # lr_union = left_mask + right_mask
        # lr_union = torch.where(lr_union == 2, one_tensor, lr_union)
        # lr_intersection = left_mask * right_mask
        # lr_union = torch.sum(lr_union, dim=(1,2))
        # lr_intersection = torch.sum(lr_intersection, dim=(1,2))
        # lr_mask_iou = lr_intersection / (lr_union + 1e-8)

        bg_mask = torch.where(predict_mask[:, 5, :, :] > 0.5, one_tensor, zero_tensor)
        person_mask = torch.where(predict_mask[:, 0, :, :] > 0.5, one_tensor, zero_tensor)
        whole_union = bg_mask + person_mask
        whole_union = torch.where(whole_union == 2, one_tensor, whole_union)
        whole_intersection = bg_mask * person_mask
        whole_union = torch.sum(whole_union, dim=(1,2))
        whole_intersection = torch.sum(whole_intersection, dim=(1,2))
        whole_mask_iou =  whole_intersection / (whole_union + 1e-8)

        knee_mask = torch.where(predict_mask[:, 3, :, :] > 0.5, one_tensor, zero_tensor)
        feet_mask = torch.where(predict_mask[:, 4, :, :] > 0.5, one_tensor, zero_tensor)
        lower_mask =  torch.where(predict_mask[:, 2, :, :] > 0.5, one_tensor, zero_tensor)

        feet_lower_intersection = feet_mask * lower_mask
        knee_lower_intersection = knee_mask * lower_mask

        
        # loss = 0.0001 * self.criterion(lr_mask_iou, torch.zeros(predict_mask.shape[0]).cuda()).mean()
        loss = 0.001 * self.criterion(feet_lower_intersection, feet_mask).mean()
        loss += 0.001 * self.criterion(knee_lower_intersection, knee_mask).mean()
        loss += 0.001 * self.criterion(whole_mask_iou, torch.zeros(predict_mask.shape[0]).cuda()).mean()

        return loss


class AdaptiveLoss(nn.Module):
    def __init__(self, ignore_label=-1, weight=None):
        super(MasksCrossEntropy, self).__init__()
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(weight=weight,
                                             ignore_index=ignore_label)

    # left 交 right = 0
    # upper 交 lower = 0
    # upper 并 lower = left 并 right = whole body
    # 315 < orientation < 45 度, left -> right 
    def forward(self, scores, targets):
        loss = 0
        for i in range(5):
            score = scores[:,i,:,:]
            target = targets[:,i,:,:]
            ph, pw = score.size(1), score.size(2)
            h, w = target.size(1), target.size(2)
            if ph != h or pw != w:
                score = F.upsample(input=score, size=(h, w), mode='bilinear', align_corners=True)

            # import pdb;pdb.set_trace()
            if target.min() == -1:
                loss += 0
                # print("none mask label")
            else:
                loss += 0.000001 * self.criterion(score, target)
                # print("mask loss")
        
        return loss


class MasksMSELoss(nn.Module):
    def __init__(self, weight=None):
        super(MasksMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='mean')

    def forward(self, masks, targets):
        loss = 0.0
        batch_size = masks.size(0)
        num_channels = masks.size(1)
        masks = masks.float().reshape(batch_size, num_channels, -1)
        targets = targets.float().reshape(batch_size, num_channels, -1)
        for i in range(batch_size):
            mask = masks[i,:,:]
            target = targets[i,:,:]
            if target.max() == -1:
                loss += 0.0
            else:
                for idx in range(num_channels):
                    # import pdb;pdb.set_trace()
                    heatmap_pred = mask[idx]
                    heatmap_gt = target[idx]
                    loss += 0.00005 * self.criterion(heatmap_pred, heatmap_gt)
        return loss

class JointsOHKMMSELoss(nn.Module):
    def __init__(self, use_target_weight, topk=8):
        super(JointsOHKMMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='none')
        self.use_target_weight = use_target_weight
        self.topk = topk

    def ohkm(self, loss):
        ohkm_loss = 0.
        for i in range(loss.size()[0]):
            sub_loss = loss[i]
            topk_val, topk_idx = torch.topk(
                sub_loss, k=self.topk, dim=0, sorted=False
            )
            tmp_loss = torch.gather(sub_loss, 0, topk_idx)
            ohkm_loss += torch.sum(tmp_loss) / self.topk
        ohkm_loss /= loss.size()[0]
        return ohkm_loss

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)

        loss = []
        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                loss.append(0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                ))
            else:
                loss.append(
                    0.5 * self.criterion(heatmap_pred, heatmap_gt)
                )

        loss = [l.mean(dim=1).unsqueeze(dim=1) for l in loss]
        loss = torch.cat(loss, dim=1)

        return self.ohkm(loss)


def _tranpose_and_gather_scalar(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, 1)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), feat.size(2))
    feat = feat.gather(1, ind)
    return feat


def reg_l1_loss(pred, target ,mask, has_3d_label):
    pred = torch.squeeze(pred)
    target = torch.squeeze(target)
    loss = torch.abs(pred - target) * mask
    loss = loss.sum()
    num = mask.float().sum()
    loss = loss / (num + 1e-4)
    return loss

class DepthLoss(nn.Module):
    def __init__(self):
        super(DepthLoss, self).__init__()
    def forward(self, output, mask, ind, target, has_3d_label):
        # pay attention that all the variables are depth
        pred = _tranpose_and_gather_scalar(output, ind)
        loss = reg_l1_loss(pred, target ,mask, has_3d_label)
        return loss

class hoe_diff_loss(nn.Module):
    def __init__(self):
        super(hoe_diff_loss, self).__init__()
        # self.softmax_layer = nn.Softmax2d()
        self.compute_norm = nn.L1Loss()

    def forward(self, plane_output, depth_output, hoe_output_val, ind, has_hoe_label):
        # get the width value using intergal
        # plane_output_softmax = self.softmax_layer(plane_output)
        H, W = plane_output.shape[2:]
        plane_output_softmax = plane_output.reshape(plane_output.shape[0], plane_output.shape[1], -1)
        part_sum = plane_output_softmax.sum(2, keepdim=True)
        plane_output_softmax = plane_output_softmax / part_sum
        plane_output_softmax = plane_output_softmax.reshape(plane_output_softmax.shape[0], plane_output_softmax.shape[1], H, W)

        interg_H = torch.sum(plane_output_softmax, 2, keepdim=False)
        W_value = interg_H * torch.arange(1, interg_H.shape[-1]+1).type(torch.cuda.FloatTensor)
        W_value = torch.sum(W_value, 2)
        W_value /= min(H, W)

        # get the depth value
        pred_h = _tranpose_and_gather_scalar(depth_output, ind)
        pred_h = torch.squeeze(pred_h)

        # compute hoe of 3D pose coordinates
        # hoe_from_3d = torch.atan2(pred_h[:, 14] - pred_h[:, 11],
        #                           W_value[:, 14] - W_value[:, 11]) / 3.1415926 * 180 / 5
        hoe_from_3d = torch.atan2(pred_h[:, 14] - pred_h[:, 11],
                                  W_value[:, 14] - W_value[:, 11])
        sin_angle = torch.sin(hoe_from_3d)
        cos_angle = torch.cos(hoe_from_3d)

        sin_gt = torch.sin(hoe_output_val)
        cos_gt = torch.cos(hoe_output_val)
        loss = ((sin_angle - sin_gt) ** 2 + (cos_angle - cos_gt) ** 2)
        loss = loss * has_hoe_label
        loss = loss.sum() / (has_hoe_label.sum() + 1e-4)
        # compute the distance
        # loss = (36 - torch.abs(torch.abs(hoe_output_val - hoe_from_3d) - 36)).mean()
        # loss = self.compute_norm(hoe_from_3d, hoe_output_val)
        return loss

class Bone_loss(nn.Module):
    def __init__(self):
        super(Bone_loss, self).__init__()
    def forward(self, output, mask, ind, target, gt_2d):
        pred = _tranpose_and_gather_scalar(output, ind)
        bone_func = VarLoss(1)
        loss = bone_func(pred, target, mask, gt_2d)
        return loss

class Vis_loss(nn.Module):
    def __init__(self):
        super(Vis_loss, self).__init__()
        self.criterion = nn.CrossEntropyLoss(reduction='none')
    def forward(self, joint_vis, joint_gt):
        # import pdb;pdb.set_trace()

        joint_gt = joint_gt.squeeze(2)
        loss_tmp = self.criterion(joint_vis,joint_gt)
        loss = 0
        for i in range(len(loss_tmp)):
            loss += 0.1 * loss_tmp[i]
        return 0.001 * loss

class Weight_loss(nn.Module):
    def __init__(self):
        super(Weight_loss, self).__init__()
        # self.criterion = nn.MSELoss(reduction='none')
        self.criterion = nn.CrossEntropyLoss(reduction='none')


    def forward(self, weight, weight_gt):
        weight_gt = weight_gt.squeeze()
        loss = self.criterion(weight, weight_gt)
        # import pdb;pdb.set_trace()
        # for i in range(len(loss_tmp)):
        #     loss += 0.001 * loss_tmp[i]
        return 0.001 * loss.mean()

class VarLoss(Function):
    def __init__(self, var_weight):
        super(VarLoss, self).__init__()
        self.var_weight = var_weight
        # self.skeleton_idx = [[[0, 1], [1, 2],
        #                       [3, 4], [4, 5]],
        #                      [[10, 11], [11, 12],
        #                       [13, 14], [14, 15]],
        #                      [[2, 6], [3, 6]],
        #                      [[12, 8], [13, 8]]]
        # [0, 1, 2, 3, 4, 5, 6, 7, 8,  9, 10, 11, 12, 13, 14, 15]
        # [3, 2, 1, 4, 5, 6, 0, 7, 8, 10, 16, 15, 14, 11, 12, 13]
        self.skeleton_idx = [[[3, 2], [2, 1],
                              [4, 5], [5, 6]],
                             [[16, 15], [15, 14],
                              [11, 12], [12, 13]],
                             [[1, 0], [4, 0]],
                             [[14, 8], [11, 8]]]
        self.skeleton_weight = [[1.0085885098415446, 1,
                                 1, 1.0085885098415446],
                                [1.1375361376887123, 1,
                                 1, 1.1375361376887123],
                                [1, 1],
                                [1, 1]]

    def forward(self, input, visible, mask, gt_2d):
        xy = gt_2d.view(gt_2d.size(0), -1, 2)
        batch_size = input.size(0)
        output = torch.cuda.FloatTensor(1) * 0
        for t in range(batch_size):
            if mask[t].sum() == 0:  # mask is the mask for supervised depth
                # xy[t] = 2.0 * xy[t] / ref.outputRes - 1
                for g in range(len(self.skeleton_idx)):
                    E, num = 0, 0
                    N = len(self.skeleton_idx[g])
                    l = np.zeros(N)
                    for j in range(N):
                        id1, id2 = self.skeleton_idx[g][j]
                        if visible[t, id1] > 0.5 and visible[t, id2] > 0.5:
                            l[j] = (((xy[t, id1] - xy[t, id2]) ** 2).sum() + \
                                    (input[t, id1] - input[t, id2]) ** 2) ** 0.5
                            l[j] = l[j] * self.skeleton_weight[g][j]
                            num += 1
                            E += l[j]
                    if num < 0.5:
                        E = 0
                    else:
                        E = E / num
                    loss = 0
                    for j in range(N):
                        if l[j] > 0:
                            loss += (l[j] - E) ** 2 / 2. / num
                    output += loss
        output = self.var_weight * output / batch_size
        self.save_for_backward(input, visible, mask, gt_2d)
        # output = output.cuda(self.device, non_blocking=True)
        return output

    def backward(self, grad_output):
        input, visible, mask, gt_2d = self.saved_tensors
        xy = gt_2d.view(gt_2d.size(0), -1, 2)
        grad_input = torch.zeros(input.size()).type(torch.cuda.FloatTensor)
        batch_size = input.size(0)
        for t in range(batch_size):
            if mask[t].sum() == 0:  # mask is the mask for supervised depth
                for g in range(len(self.skeleton_idx)):
                    E, num = 0, 0
                    N = len(self.skeleton_idx[g])
                    l = np.zeros(N)
                    for j in range(N):
                        id1, id2 = self.skeleton_idx[g][j]
                        if visible[t, id1] > 0.5 and visible[t, id2] > 0.5:
                            l[j] = (((xy[t, id1] - xy[t, id2]) ** 2).sum() + \
                                    (input[t, id1] - input[t, id2]) ** 2) ** 0.5
                            l[j] = l[j] * self.skeleton_weight[g][j]
                            num += 1
                            E += l[j]
                    if num < 0.5:
                        E = 0
                    else:
                        E = E / num
                    for j in range(N):
                        if l[j] > 0:
                            id1, id2 = self.skeleton_idx[g][j]
                            grad_input[t][id1] += self.var_weight * \
                                                  self.skeleton_weight[g][j] ** 2 / num * (l[j] - E) \
                                                  / l[j] * (input[t, id1] - input[t, id2]) / batch_size
                            grad_input[t][id2] += self.var_weight * \
                                                  self.skeleton_weight[g][j] ** 2 / num * (l[j] - E) \
                                                  / l[j] * (input[t, id2] - input[t, id1]) / batch_size
        return grad_input, None, None, None


class Ori_Constraint_Loss(nn.Module):
    def __init__(self, use_target_weight):
        super(Ori_Constraint_Loss, self).__init__()
        self.criterion = nn.MSELoss(reduction='mean')
        self.use_target_weight = use_target_weight

    def forward(self, output, degree):
        batch_size = output.size(0)
        num_parts = output.size(1)
        loss = 0

        
        for idx in range(num_parts):
            heatmap_pred = output[idx].squeeze()
            heatmap_gt = degree[idx].squeeze()
            if self.use_target_weight:
                loss += 0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                )
            else:
                loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)

        return loss / num_joints



class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """
    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        loss = 1 - num / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))


class DiceLoss(nn.Module):
    """Dice loss, need one hot encode input
    Args:
        weight: An array of shape [num_classes,]
        ignore_index: class index to ignore
        predict: A tensor of shape [N, C, *]
        target: A tensor of same shape with predict
        other args pass to BinaryDiceLoss
    Return:
        same as BinaryDiceLoss
    """
    def __init__(self, weight=None, ignore_index=None, **kwargs):
        super(DiceLoss, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, predict, target, mask_code):
        ph, pw = predict.size(2), predict.size(3)
        h, w = target.size(2), target.size(3)
        if ph != h or pw != w:
            predict = F.upsample(input=predict, size=(h, w), mode='bilinear', align_corners=True)
        assert predict.shape == target.shape, 'predict & target shape do not match'

        dice = BinaryDiceLoss(**self.kwargs)
        total_loss = 0
        predict = predict[mask_code==1]
        target = target[mask_code==1]
        for i in range(target.shape[1]):
            if i != self.ignore_index:
                dice_loss = dice(predict[:, i], target[:, i])
                if self.weight is not None:
                    assert self.weight.shape[0] == target.shape[1], \
                        'Expect weight shape [{}], get[{}]'.format(target.shape[1], self.weight.shape[0])
                    dice_loss *= self.weight[i]
                total_loss += dice_loss
        return 0.001 * total_loss/target.shape[1]

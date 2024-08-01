from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import cv2
import shutil
from PIL import Image
import matplotlib.pyplot as plt
from lib.core.inference import get_max_preds
import gc
from lib.utils.utils import vh2hoe, bin2hoe

def calc_dists(preds, target, normalize):
    preds = preds.astype(np.float32)
    target = target.astype(np.float32)
    dists = np.zeros((preds.shape[1], preds.shape[0]))
    for n in range(preds.shape[0]):
        for c in range(preds.shape[1]):
            if target[n, c, 0] > 1 and target[n, c, 1] > 1:
                normed_preds = preds[n, c, :] / normalize[n]
                normed_targets = target[n, c, :] / normalize[n]
                dists[c, n] = np.linalg.norm(normed_preds - normed_targets)
            else:
                dists[c, n] = -1
    return dists


def dist_acc(dists, thr=0.5):
    ''' Return percentage below threshold while ignoring values with a -1 '''
    dist_cal = np.not_equal(dists, -1)
    num_dist_cal = dist_cal.sum()
    if num_dist_cal > 0:
        return np.less(dists[dist_cal], thr).sum() * 1.0 / num_dist_cal
    else:
        return -1
def get_offset(hm_hps, idx):
    """
    Get offset according to idx and clip it
    :param hm_hps:
    :param idx:
    :return: offset array [batch, joints, 2]
    """
    n_b, n_j, _ = idx.shape
    offset = np.empty_like(idx)
    for i in range(n_b):
        for j in range(n_j):
            offset[i, j, :] = hm_hps[i, 2*j:2*j+2, int(idx[i, j, 1]), int(idx[i, j, 0])]
            # offset[i, j, 0] = hm_hps[i, 2*j, int(idx[i, j, 1]), int(idx[i, j, 0])]
            # offset[i, j, 1] = hm_hps[i, 2*j+1, int(idx[i, j, 1]), int(idx[i, j, 0])]
    return offset

def dist(offset_preds, offset_targets, target, norm):
    """
    rewrite calc_dist()
    :param offset_preds:
    :param offset_targets:
    :return:
    """
    mask = np.greater(target, 0).sum(axis=2)
    mask = np.greater(mask, 1)
    if norm.ndim != offset_targets.ndim or norm.ndim != offset_preds.ndim:
        norm = np.expand_dims(norm, axis=1)
    norm_offset_preds = offset_preds / norm
    norm_offset_targets = offset_targets / norm
    tmp = norm_offset_preds - norm_offset_targets
    dists = np.linalg.norm(tmp, axis=2)
    non_dists = -1 * np.ones_like(dists)
    dists = dists * mask + non_dists * (1-mask)
    return dists.transpose(1, 0)  # [joints, batch]

def accuracy2(output, hm_hps, target, target_hm_hps, locref_stdev, thr=0.5):
    num_joints = output.shape[1]
    h = output.shape[2]
    w = output.shape[3]
    norm = np.ones((output.shape[0], 2)) * np.array([h, w]) / 10

    int_pred, _ = get_max_preds(output)  # [batch, joint, 2]
    int_target, _ = get_max_preds(target)
    offset_pred = get_offset(hm_hps, int_pred)  # [batch, joints, 2]
    offset_target = get_offset(target_hm_hps, int_target)

    pred   = int_pred + offset_pred*locref_stdev
    target = int_target + offset_target*locref_stdev
    offset_norm = np.ones((pred.shape[0], 2))
    final_norm = norm

    int_dists = calc_dists(int_pred, int_target, norm)
    offset_dists = dist(offset_pred, offset_target, int_target, offset_norm)
    dists = dist(pred, target, int_target, final_norm)

    acc = np.zeros((num_joints + 1, 3))
    avg_acc = np.zeros(3)
    cnt = np.zeros(3)

    for i in range(num_joints):
        acc[i + 1, 0] = dist_acc(dists[i], thr)
        acc[i + 1, 1] = dist_acc(int_dists[i], thr)
        acc[i + 1, 2] = dist_acc(offset_dists[i], thr)
        if acc[i + 1, 0] > 0:
            avg_acc[0] += acc[i + 1, 0]
            cnt[0] += 1
        if acc[i + 1, 1] > 0:
            avg_acc[1] += acc[i + 1, 1]
            cnt[1] += 1
        if acc[i + 1, 2] > 0:
            avg_acc[2] += acc[i + 1, 2]
            cnt[2] += 1

    for j in range(3):
        if cnt[j] != 0:
            avg_acc[j] = avg_acc[j] / cnt[j]
            acc[0, j] = avg_acc[j]
    return acc, avg_acc, cnt, pred



def accuracy(output, target, hm_type='gaussian', thr=0.5):
    '''
    Calculate accuracy according to PCK,
    but uses ground truth heatmap rather than x,y locations
    First value to be returned is average accuracy across 'idxs',
    followed by individual accuracies
    '''
    idx = list(range(output.shape[1]))
    norm = 1.0
    if hm_type == 'gaussian':
        pred, _ = get_max_preds(output)
        target, _ = get_max_preds(target)
        h = output.shape[2]
        w = output.shape[3]
        norm = np.ones((pred.shape[0], 2)) * np.array([h, w]) / 10
    dists = calc_dists(pred, target, norm)

    acc = np.zeros((len(idx) + 1))
    avg_acc = 0
    cnt = 0

    for i in range(len(idx)):
        acc[i + 1] = dist_acc(dists[idx[i]])
        if acc[i + 1] >= 0:
            avg_acc = avg_acc + acc[i + 1]
            cnt += 1

    avg_acc = avg_acc / cnt if cnt != 0 else 0
    if cnt != 0:
        acc[0] = avg_acc
    return acc, avg_acc, cnt, pred

def continous_comp_deg_error(output, gt_degree):
    result = 0
    index_degree = output.argmax(axis = 1)
    excellent = 0
    mid = 0
    poor_225 = 0
    poor = 0
    poor_45 = 0
    for i in range(len(index_degree)):
        diff = abs(index_degree[i]*5 - gt_degree[i])
        diff = min(diff, 360 - diff)
        result += diff
        if diff <= 45:
            poor_45 += 1
            if diff <= 30:
                poor += 1
                if diff <= 22.5:
                    poor_225 += 1
                    if diff <= 15:
                        mid += 1
                        if diff <= 5:
                            excellent += 1
    return result/len(output), excellent, mid, poor_225, poor, poor_45, gt_degree, index_degree*5, len(output)

def bin_comp_deg_error(vertical, horizontal, vertical_gt, horizontal_gt):
    result = 0
    horizontal = horizontal.argmax(axis = 1)
    horizontal_gt = horizontal_gt.argmax(axis = 1)

    pre_degree = bin2hoe(vertical, horizontal)
    degree = bin2hoe(vertical_gt, horizontal_gt)
    # index_degree is the prediction
    # index_degree = output.argmax(axis = 1)
    excellent = 0
    mid = 0
    poor_225 = 0
    poor = 0
    poor_45 = 0
    for i in range(len(pre_degree)):
        diff = abs(pre_degree[i] - degree[i]) * 5
        diff = min(diff, 360 - diff)
        result += diff
        if diff <= 45:
            poor_45 += 1
            if diff <= 30:
                poor += 1
                if diff <= 22.5:
                    poor_225 += 1
                    if diff <= 15:
                        mid += 1
                        if diff <= 5:
                            excellent += 1
    return result/len(vertical), excellent, mid, poor_225, poor, poor_45,degree*5 ,pre_degree * 5, len(vertical)

def vh_comp_deg_error(vertical, horizontal, vertical_gt, horizontal_gt):
    result = 0
    vertical = vertical.argmax(axis = 1)
    horizontal = horizontal.argmax(axis = 1)
    vertical_gt = vertical_gt.argmax(axis = 1)
    horizontal_gt = horizontal_gt.argmax(axis = 1)

    index_degree = vh2hoe(vertical, horizontal)
    degree = vh2hoe(vertical_gt, horizontal_gt)
    # index_degree is the prediction
    # index_degree = output.argmax(axis = 1)
    excellent = 0
    mid = 0
    poor_225 = 0
    poor = 0
    poor_45 = 0
    for i in range(len(index_degree)):
        diff = abs(index_degree[i] - degree[i]) * 5
        diff = min(diff, 360 - diff)
        result += diff
        if diff <= 45:
            poor_45 += 1
            if diff <= 30:
                poor += 1
                if diff <= 22.5:
                    poor_225 += 1
                    if diff <= 15:
                        mid += 1
                        if diff <= 5:
                            excellent += 1
    return result/len(vertical), excellent, mid, poor_225, poor, poor_45,degree*5 ,index_degree * 5, len(vertical)

def comp_deg_error(output, degree):
    result = 0
    degree = degree.argmax(axis = 1)
    # index_degree is the prediction
    index_degree = output.argmax(axis = 1)
    excellent = 0
    mid = 0
    poor_225 = 0
    poor = 0
    poor_45 = 0
    poor_90 = 0
    for i in range(len(index_degree)):
        diff = abs(index_degree[i] - degree[i]) * 5
        diff = min(diff, 360 - diff)
        result += diff
        if diff <=90:
            poor_90 += 1
            if diff <= 45:
                poor_45 += 1
                if diff <= 30:
                    poor += 1
                    if diff <= 22.5:
                        poor_225 += 1
                        if diff <= 15:
                            mid += 1
                            if diff <= 5:
                                excellent += 1
    return result/len(output), excellent, mid, poor_225, poor, poor_45, poor_90,degree*5 ,index_degree * 5, len(output)

def comp_test_deg_error(output, degree):
    result = 0
    # index_degree is the prediction
    index_degree = output.argmax(axis = 1)
    excellent = 0
    mid = 0
    poor_225 = 0
    poor = 0
    poor_45 = 0
    poor_90 = 0
    for i in range(len(index_degree)):
        diff = abs(index_degree[i] - degree[i]) * 5
        diff = min(diff, 360 - diff)
        result += diff
        if diff <=90:
            poor_90 += 1
            if diff <= 45:
                poor_45 += 1
                if diff <= 30:
                    poor += 1
                    if diff <= 22.5:
                        poor_225 += 1
                        if diff <= 15:
                            mid += 1
                            if diff <= 5:
                                excellent += 1
    return result/len(output), excellent, mid, poor_225, poor, poor_45, poor_90,degree*5 ,index_degree * 5, len(output)


# this is for human3.6
def mpjpe(heatmap, depthmap, gt_3d, convert_func):
  preds_3d = get_preds_3d(heatmap, depthmap)
  cnt, pjpe = 0, 0
  for i in range(preds_3d.shape[0]):
    if gt_3d[i].sum() ** 2 > 0:
      cnt += 1
      pred_3d_h36m = convert_func(preds_3d[i])
      err = (((gt_3d[i] - pred_3d_h36m) ** 2).sum(axis=1) ** 0.5).mean()
      pjpe += err
  if cnt > 0:
    pjpe /= cnt
  return pjpe, cnt

def get_preds_3d(heatmap, depthmap):
  output_res = min(heatmap.shape[2], heatmap.shape[3])
  preds = get_preds(heatmap).astype(np.int32)
  preds_3d = np.zeros((preds.shape[0], preds.shape[1], 3), dtype=np.float32)
  for i in range(preds.shape[0]):
    for j in range(preds.shape[1]):
      idx = min(j, depthmap.shape[1] - 1)
      pt = preds[i, j]
      preds_3d[i, j, 2] = depthmap[i, idx, pt[1], pt[0]]
      preds_3d[i, j, :2] = 1.0 * preds[i, j] / output_res
    preds_3d[i] = preds_3d[i] - preds_3d[i, 0:1]
  return preds_3d

def get_preds(hm, return_conf=False):
    assert len(hm.shape) == 4, 'Input must be a 4-D tensor'
    h = hm.shape[2]
    w = hm.shape[3]
    hm = hm.reshape(hm.shape[0], hm.shape[1], hm.shape[2] * hm.shape[3])
    idx = np.argmax(hm, axis=2)

    preds = np.zeros((hm.shape[0], hm.shape[1], 2))
    for i in range(hm.shape[0]):
        for j in range(hm.shape[1]):
            preds[i, j, 0], preds[i, j, 1] = idx[i, j] % w, idx[i, j] / w
    if return_conf:
        conf = np.amax(hm, axis=2).reshape(hm.shape[0], hm.shape[1], 1)
        return preds, conf
    else:
        return preds

# this is to  draw images
def draw_orientation(img_np, gt_ori, pred_ori , path, alis=''):
    for idx in range(len(gt_ori)):
        img_tmp = img_np[idx]
        img_tmp = np.transpose(img_tmp, axes=[1, 2, 0])
        img_tmp *= [0.229, 0.224, 0.225]
        img_tmp += [0.485, 0.456, 0.406]
        img_tmp *= 255.0
        img_tmp = img_tmp.astype(np.uint8)

        # then draw the image
        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_subplot(1, 1, 1)

        theta_1 = gt_ori[idx]/180 * np.pi + np.pi/2
        theta_2 = pred_ori[idx]/180 * np.pi + np.pi/2
        plt.plot([0, np.cos(theta_1)], [0, np.sin(theta_1)], color="red", linewidth=3)
        plt.plot([0, np.cos(theta_2)], [0, np.sin(theta_2)], color="blue", linewidth=3)
        circle = plt.Circle((0, 0), 1, color='black', fill=False, linewidth=2)
        ax.add_patch(circle)
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.axis('off')
        fig.savefig(os.path.join(path, str(idx)+'_'+alis+'.jpg'))
        ori_img = cv2.imread(os.path.join(path, str(idx)+'_'+alis+'.jpg'))

        width = img_tmp.shape[1]
        ori_img = cv2.resize(ori_img, (width, width), interpolation=cv2.INTER_CUBIC)
        img_all = np.concatenate([img_tmp, ori_img],axis=0)
        im = Image.fromarray(img_all)
        im.save(os.path.join(path, str(idx)+'_'+alis+'_raw.jpg'))
        plt.close("all")
        del ori_img,img_all, im,img_tmp
        gc.collect()


def ori_numpy(gt_ori, pred_ori):
    ori_list = []
    for idx in range(len(gt_ori)):
        ori_list.append((gt_ori[idx], pred_ori[idx]))
    return ori_list










from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
 
import time
import logging
import math
import os
from turtle import pd
import matplotlib.pyplot as plt
from PIL import Image
import gc
import numpy as np
import torch
import pickle
import cv2
from torch.autograd import Variable
import torch.nn.functional as F


from core.evaluate import accuracy, accuracy2
from core.evaluate import comp_deg_error, continous_comp_deg_error, ori_numpy, vh_comp_deg_error, bin_comp_deg_error
from lib.utils.utils import vh2hoe, get_cos_similar_multi, bin2hoe

logger = logging.getLogger(__name__)

def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)
def vis_frame(frame, keypoints, format='coco'):
    '''
    frame: frame image
    im_res: im_res of predictions
    format: coco or mpii

    return rendered image
    '''
    BLUE = (255, 0, 0)

    kp_num = 23

    if kp_num == 17:
        l_pair = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Head
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
            (17, 11), (17, 12),  # Body
            (11, 13), (12, 14), (13, 15), (14, 16)
        ]

        p_color = [(0, 255, 255), (0, 191, 255), (0, 255, 102), (0, 77, 255), (0, 255, 0),  # Nose, LEye, REye, LEar, REar
                    (77, 255, 255), (77, 255, 204), (77, 204, 255), (191, 255, 77), (77, 191, 255), (191, 255, 77),  # LShoulder, RShoulder, LElbow, RElbow, LWrist, RWrist
                    (204, 77, 255), (77, 255, 204), (191, 77, 255), (77, 255, 191), (127, 77, 255), (77, 255, 127), (0, 255, 255)]  # LHip, RHip, LKnee, Rknee, LAnkle, RAnkle, Neck
        line_color = [(0, 215, 255), (0, 255, 204), (0, 134, 255), (0, 255, 50),
                        (77, 255, 222), (77, 196, 255), (77, 135, 255), (191, 255, 77), (77, 255, 77),
                        (77, 222, 255), (255, 156, 127),
                        (0, 127, 255), (255, 127, 77), (0, 77, 255), (255, 77, 36)]
    elif kp_num == 23:
        l_pair = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Head
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
            (23, 11), (23, 12),  # Body
            (11, 13), (12, 14), (13, 15),(15,17), (15,18),(15,19), (14, 16), (16,20), (16,21),(16,22)
        ]

        p_color = [(0, 255, 255), (0, 191, 255), (0, 255, 102), (0, 77, 255), (0, 255, 0),  # Nose, LEye, REye, LEar, REar
                    (77, 255, 255), (77, 255, 204), (77, 204, 255), (191, 255, 77), (77, 191, 255), (191, 255, 77),  # LShoulder, RShoulder, LElbow, RElbow, LWrist, RWrist
                    (204, 77, 255), (77, 255, 204), (191, 77, 255), (77, 255, 191), (127, 77, 255), (77, 255, 127),  (127, 77, 255), (127, 77, 255), (127, 77, 255), (77, 255, 127), (77, 255, 127),(77, 255, 127),(0, 255, 255)]  # LHip, RHip, LKnee, Rknee, LAnkle, RAnkle, Neck
        line_color = [(0, 215, 255), (0, 255, 204), (0, 134, 255), (0, 255, 50),
                        (77, 255, 222), (77, 196, 255), (77, 135, 255), (191, 255, 77), (77, 255, 77),
                        (77, 222, 255), (255, 156, 127),
                        (0, 127, 255), (255, 127, 77), (0, 77, 255), (0, 77, 255),(0, 77, 255),(0, 77, 255), (255, 77, 36), (255, 77, 36),(255, 77, 36),(255, 77, 36)]
    # im_name = os.path.basename(im_res['imgname'])
    img = frame.copy()
    height, width = img.shape[:2]
    part_line = {}
    # import pdb;pdb.set_trace()
    tmp = np.array(list(keypoints.values()))
    kp_preds = tmp[:,:2]
    kp_scores = tmp[:,2]
    if kp_num == 17:
        kp_preds = np.concatenate([kp_preds, np.expand_dims( (kp_preds[5] + kp_preds[6]) / 2,axis=0) ], axis=0)
        kp_scores = np.concatenate([kp_scores, np.expand_dims( (kp_scores[5] + kp_scores[6])/ 2,axis=0) ], axis=0)
        color = BLUE
    else:
        kp_preds = np.concatenate([kp_preds, np.expand_dims( (kp_preds[5] + kp_preds[6]) / 2,axis=0) ], axis=0)
        kp_scores = np.concatenate([kp_scores, np.expand_dims( (kp_scores[5] + kp_scores[6])/ 2,axis=0) ], axis=0)
        color = BLUE
        kp_preds
    # Draw keypoints
    for n in range(kp_scores.shape[0]):
        if kp_scores[n] <= 0.2:
            continue
        cor_y, cor_x  = int(kp_preds[n, 0]), int(kp_preds[n, 1])
        part_line[n] = (int(cor_x*4), int(cor_y*4))
        bg = img.copy()
        # if n < len(p_color):
            # cv2.circle(bg, (int(cor_x*8), int(cor_y*8)), 2, p_color[n], -1)
        # else:
            # cv2.circle(bg, (int(cor_x*8), int(cor_y*8)), 1, (255,255,255), 2)
        cv2.circle(bg, (int(cor_x*4), int(cor_y*4)), 3, (255,0,0), 2)
        # Now create a mask of logo and create its inverse mask also
        # if n < len(p_color):
        #     transparency = float(max(0, min(1, kp_scores[n])))
        # else:
        #     transparency = float(max(0, min(1, kp_scores[n]*2)))
        transparency = 1.0
        img = cv2.addWeighted(bg, transparency, img, 1 - transparency, 0)
    # Draw limbs
    for i, (start_p, end_p) in enumerate(l_pair):
        print(i)
        if start_p in part_line and end_p in part_line:
            
            start_xy = part_line[start_p]
            end_xy = part_line[end_p]
            print(start_xy, end_xy)
            bg = img.copy()

            X = (start_xy[0], end_xy[0])
            Y = (start_xy[1], end_xy[1])
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((Y[0] - Y[1]) ** 2 + (X[0] - X[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(Y[0] - Y[1], X[0] - X[1]))
            stickwidth = (kp_scores[start_p] + kp_scores[end_p]) + 1
            polygon = cv2.ellipse2Poly((int(mX), int(mY)), (int(length/2), int(stickwidth)), int(angle), 0, 360, 1)
            if i < len(line_color):
                cv2.fillConvexPoly(bg, polygon, line_color[i])
            else:
                cv2.line(bg, start_xy, end_xy, (255,255,255), 1)
            # if n < len(p_color):
            #     transparency = float(max(0, min(1, 0.5 * (kp_scores[start_p] + kp_scores[end_p])-0.1)))
            # else:
            #     transparency = float(max(0, min(1, (kp_scores[start_p] + kp_scores[end_p]))))
            transparency = 1.0

            #transparency = float(max(0, min(1, 0.5 * (kp_scores[start_p] + kp_scores[end_p])-0.1)))
            img = cv2.addWeighted(bg, transparency, img, 1 - transparency, 0)
    return img


def draw_conf_orientation(img_np, gt_ori, pred_ori ,path, keypoints, confidence, alis=''):
    if not os.path.exists(path):
        os.makedirs(path)
    # then draw the image
    for idx in range(len(pred_ori)):
        img_tmp = img_np[idx]

        img_tmp = np.transpose(img_tmp, axes=[1, 2, 0])
        img_tmp *= [0.229, 0.224, 0.225]
        img_tmp += [0.485, 0.456, 0.406]
        img_tmp *= 255.0
        img_tmp = img_tmp.astype(np.uint8)
        img_tmp = vis_frame(img_tmp, keypoints[idx])

        fig = plt.figure(figsize=(2, 2))
        ax = fig.add_subplot(1, 1, 1)

        theta_1 = gt_ori[idx]/180 * np.pi + np.pi/2
        plt.plot([0, np.cos(theta_1)], [0, np.sin(theta_1)], color="red", linewidth=3)
        theta_2 = pred_ori[idx]/180 * np.pi + np.pi/2
        plt.plot([0, np.cos(theta_2)], [0, np.sin(theta_2)], color="blue", linewidth=3)
        circle = plt.Circle((0, 0), 1, color='black', fill=False, linewidth=2)
        ax.add_patch(circle)
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.axis('off')
        # plt.text(0.5, 0.01, '{:.2f}'.format(confidence[idx][0]), ha='center', va='bottom', transform=plt.gcf().transFigure)
        plt.text(0.5, 0.01, '{:.2f}'.format(confidence[idx]), ha='center', va='bottom', transform=plt.gcf().transFigure)

        fig.savefig(os.path.join(path, str(idx)+'_'+alis+'.jpg'))
        ori_img = cv2.imread(os.path.join(path, str(idx)+'_'+alis+'.jpg'))
        ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)

        width = img_tmp.shape[1]
        ori_img = cv2.resize(ori_img, (width, width), interpolation=cv2.INTER_CUBIC)
        img_all = np.concatenate([img_tmp, ori_img],axis=0)
        im = Image.fromarray(img_all)
        im.save(os.path.join(path, str(idx)+'_'+alis+'_raw.jpg'))
        plt.close("all")
        del ori_img,img_all, im,img_tmp
        gc.collect()
    
def draw_orientation(img_np, gt_ori, pred_ori, path, keypoints, gt_keypoints, predict_mask, alis=''):
    if not os.path.exists(path):
        os.makedirs(path)
    
    # print(gt_keypoints)
    for idx in range(len(pred_ori)):
        img_tmp = img_np[idx]

        img_tmp = np.transpose(img_tmp, axes=[1, 2, 0])
        img_tmp *= [0.229, 0.224, 0.225]
        img_tmp += [0.485, 0.456, 0.406]
        img_tmp *= 255.0
        img_tmp = img_tmp.astype(np.uint8)
        for joint_index, position in keypoints[idx].items():
            # left red , right green
            # import pdb;pdb.set_trace()
            img_tmp = img_tmp.copy()
            # print("joint_index{} x{} y{}".format(joint_index, position[0]*4, position[1]*4))
            if int(joint_index) %2 ==0:
                #right
                cv2.circle(img_tmp, (int(position[1]*4), int(position[0]*4)), radius=2, color=(0,0,255), thickness=-1, lineType=cv2.LINE_AA)
            else:
                #left
                cv2.circle(img_tmp, (int(position[1]*4), int(position[0]*4)), radius=2, color=(0,255,0), thickness=-1, lineType=cv2.LINE_AA)

        for joint_index, position in gt_keypoints[idx].items():
            # left red , right green
            # import pdb;pdb.set_trace()
            img_tmp = img_tmp.copy()
            # print("joint_index{} x{} y{}".format(joint_index, position[0]*4, position[1]*4))
            if int(joint_index) %2 ==0:
                #right
                cv2.circle(img_tmp, (int(position[1]*4), int(position[0]*4)), radius=2, color=(255,0,0), thickness=-1, lineType=cv2.LINE_AA)
            else:
                #left
                cv2.circle(img_tmp, (int(position[1]*4), int(position[0]*4)), radius=2, color=(255,255,255), thickness=-1, lineType=cv2.LINE_AA)

        # then draw the image
        fig = plt.figure(figsize=(2, 2))
        ax = fig.add_subplot(1, 1, 1)

        theta_1 = gt_ori[idx]/180 * np.pi + np.pi/2
        plt.plot([0, np.cos(theta_1)], [0, np.sin(theta_1)], color="red", linewidth=3)
        theta_2 = pred_ori[idx]/180 * np.pi + np.pi/2
        plt.plot([0, np.cos(theta_2)], [0, np.sin(theta_2)], color="blue", linewidth=3)
        circle = plt.Circle((0, 0), 1, color='black', fill=False, linewidth=2)
        ax.add_patch(circle)
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.axis('off')
        fig.savefig(os.path.join(path, str(idx)+'_'+alis+'.jpg'))
        ori_img = cv2.imread(os.path.join(path, str(idx)+'_'+alis+'.jpg'))
        ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)

        # # 全身
        # for part_idx in range(len(predict_mask[0])):
        #     project_img =  np.array(Image.fromarray(img_tmp).resize((48,64)))
        #     tmp_mask = np.array([predict_mask[idx][part_idx], predict_mask[idx][part_idx], predict_mask[idx][part_idx]])
        #     tmp_mask = tmp_mask.transpose(1,2,0)
        #     project_img[tmp_mask > 0.5] = 255
        #     project_img = Image.fromarray(project_img)

        #     project_img.save(os.path.join(path, str(idx)+'_'+alis+'_raw_mask_{}.jpg'.format(part_idx)))

        # # # background
        # project_img =  np.array(Image.fromarray(img_tmp).resize((48,64)))
        # tmp_mask = np.array([predict_mask[idx][5], predict_mask[idx][5], predict_mask[idx][5]])
        # tmp_mask = tmp_mask.transpose(1,2,0)
        # project_img[tmp_mask > 0.001] = 255
        # project_img = Image.fromarray(project_img)

        # project_img.save(os.path.join(path, str(idx)+'_'+alis+'_raw_mask5.jpg'))
        

        width = img_tmp.shape[1]
        ori_img = cv2.resize(ori_img, (width, width), interpolation=cv2.INTER_CUBIC)
        img_all = np.concatenate([img_tmp, ori_img],axis=0)
        im = Image.fromarray(img_all)
        im.save(os.path.join(path, str(idx)+'_'+alis+'_raw.jpg'))
        plt.close("all")
        del ori_img,img_all, im,img_tmp
        gc.collect()

# markdown format output
def _print_name_value(name_value, full_arch_name):
    names = name_value.keys()
    values = name_value.values()
    num_values = len(name_value)
    logger.info(
        '| Arch ' +
        ' '.join(['| {}'.format(name) for name in names]) +
        ' |'
    )
    logger.info('|---' * (num_values+1) + '|')

    if len(full_arch_name) > 15:
        full_arch_name = full_arch_name[:8] + '...'
    logger.info(
        '| ' + full_arch_name + ' ' +
        ' '.join(['| {:.3f}'.format(value) for value in values]) +
         ' |'
    )


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0


def print_msg(step, loader_len, batch_time, has_hkd, loss_hkd, loss_vertical, loss_horizontal,  losses, degree_error, acc_label, acc, speed=False, epoch = None, loss_mask=None, loss_weight=None, loss_vertical_confidence=None, loss_horizontal_confidence=None):
  
  if epoch != None:
    msg = 'Epoch: [{0}][{1}/{2}]\t' \
          'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
          'Speed {speed:.1f} samples/s\t'.format(epoch,step, loader_len, batch_time=batch_time, speed = speed)
  else:
    msg = 'Test: [{0}/{1}]\t' \
        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'.format(
            step, loader_len, batch_time=batch_time)
  if has_hkd:
    # 'Loss_vis {loss_vis.val:.3e} ({loss_vis.avg:.3e})\t' \
    if loss_horizontal_confidence != None:
        msg += 'Loss_hkd {loss_hkd.val:.3e} ({loss_hkd.avg:.3e})\t' \
            'Loss_mask {loss_mask.val:.3e} ({loss_mask.avg:.3e})\t' \
            'Loss_vertical {loss_vertical.val:.3e} ({loss_vertical.avg:.3e})\t' \
            'Loss_horizontal {loss_horizontal.val:.3e} ({loss_horizontal.avg:.3e})\t' \
            'Loss_horizontal_confidence {loss_horizontal_confidence.val:.3e} ({loss_horizontal_confidence.avg:.3e})\t' \
            'Loss {loss.val:.3e} ({loss.avg:.3e})\t'.format(loss_hkd=loss_hkd, loss_mask=loss_mask,loss=losses, loss_horizontal= loss_horizontal, loss_vertical = loss_vertical, loss_horizontal_confidence=loss_horizontal_confidence)
    else:
        msg += 'Loss_hkd {loss_hkd.val:.3e} ({loss_hkd.avg:.3e})\t' \
            'Loss_mask {loss_mask.val:.3e} ({loss_mask.avg:.3e})\t' \
            'Loss_vertical {loss_vertical.val:.3e} ({loss_vertical.avg:.3e})\t' \
            'Loss_horizontal {loss_horizontal.val:.3e} ({loss_horizontal.avg:.3e})\t' \
            'Loss {loss.val:.3e} ({loss.avg:.3e})\t'.format(loss_hkd=loss_hkd,loss_mask=loss_mask, loss=losses, loss_horizontal= loss_horizontal, loss_vertical = loss_vertical)
  else:
    msg += 'Loss {loss.val:.3e} ({loss.avg:.3e})\t'.format(loss=losses)
  
  msg += 'Degree_error {Degree_error.val:.3f} ({Degree_error.avg:.3f})\t' \
        '{acc_label} {acc.val:.1%} ({acc.avg:.1%})'.format(Degree_error = degree_error, acc_label=acc_label, acc=acc)
  logger.info(msg)

def train(config, train_loader, train_dataset, model, criterions, optimizer, epoch,
          output_dir, tb_log_dir, writer_dict, lmbda):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_2d_log = AverageMeter()
    loss_vertical_log = AverageMeter()
    loss_horizontal_log = AverageMeter()
    loss_hkd_log = AverageMeter()
    loss_mask_log = AverageMeter()
    loss_iou_log = AverageMeter()
    loss_horizontal_confidence_log = AverageMeter()
    losses = AverageMeter()
    degree_error = AverageMeter()
    acc = AverageMeter()

    # switch to train mode
    model.train()
    end = time.time()
    for i, (input, target, target_weight, horizontal_degree, meta) in enumerate(train_loader):
        # compute output
        plane_output, horizontal_output, confidence  = model(input)
        optimizer.zero_grad()

        # change to cuda format
        target = target.cuda(non_blocking=True)
        target_weight = target_weight.cuda(non_blocking=True)
        horizontal_degree = horizontal_degree.cuda(non_blocking=True)

        # compute loss
        if config.LOSS.USE_ONLY_HOE:
            loss_hoe = criterions['hoe_loss'](hoe_output, degree)
            loss_2d = loss_hoe
            # loss_mask = criterions['mask_loss'](predict_mask, mask_gt)
            loss_iou = criterions["iou_loss"](predict_mask)
            loss = loss_hoe + loss_iou + loss_mask
        else:
            eps = 1e-12
            # vertical_output = vertical_output[mask_code!=2]
            confidence = torch.clamp(confidence, 0. + eps, 1. - eps)
            b = Variable(torch.bernoulli(torch.Tensor(confidence.size()).uniform_(0, 1))).cuda()
            conf_b = confidence * b + (1 - b)
            
            horizontal_output_new = horizontal_output * conf_b.expand_as(horizontal_output) + horizontal_degree * (1 - conf_b.expand_as(horizontal_degree))
            loss_confidence = torch.mean(-torch.log(confidence))
            if 0.4 > loss_confidence.item():
                lmbda = lmbda / 1.01
            elif 0.4 <= loss_confidence.item():
                lmbda = lmbda / 0.99
            loss_horizontal = criterions['hoe_loss'](horizontal_output_new, horizontal_degree)
            loss_2d = criterions['2d_pose_loss'](plane_output, target, target_weight)

            loss = loss_horizontal + loss_2d + lmbda*loss_confidence

        # print(loss_vh_constraint)
        num_images = input.size(0)
        # measure accuracy and record loss
        loss_hkd_log.update(loss_2d.item(), num_images)
        loss_horizontal_log.update(loss_horizontal.item(), num_images)
        loss_horizontal_confidence_log.update(loss_confidence.item(), num_images)
        losses.update(loss.item(), num_images)

        # compute gradient and do update step
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        if config.DATASET.DATASET == 'tud_dataset':
            avg_degree_error, _, mid, _ , _, _, _, _, cnt = continous_comp_deg_error(hoe_output.detach().cpu().numpy(),
                                               meta['val_dgree'].numpy())
            
            acc.update(mid/cnt, cnt)
            has_hkd=False
            acc_label = 'mid15'
        elif config.LOSS.USE_ONLY_HOE:
            avg_degree_error, _, mid, _ , _, _, _, _, cnt= comp_deg_error(hoe_output.detach().cpu().numpy(),
                                                   degree.detach().cpu().numpy())
            acc.update(mid/cnt, cnt)
            has_hkd=False 
            acc_label = 'mid15'
        else:
            avg_degree_error, _, mid, _ , _, _, _, _,_, cnt= comp_deg_error(horizontal_output.detach().cpu().numpy(),
                                                   horizontal_degree.detach().cpu().numpy())
            _, avg_acc, cnt, pred = accuracy(plane_output.detach().cpu().numpy(),
                                             target.detach().cpu().numpy())
            acc.update(avg_acc, cnt)
            has_hkd=True
            acc_label = 'kpd_acc'
            

        degree_error.update(avg_degree_error, input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            print_msg(epoch = epoch, step=i, speed=input.size(0) / batch_time.val, has_hkd= has_hkd, loader_len=len(train_loader), batch_time=batch_time, loss_hkd=loss_hkd_log, loss_vertical=loss_vertical_log, loss_horizontal=loss_horizontal_log, loss_mask=loss_mask_log, losses=losses, degree_error=degree_error, acc_label=acc_label, acc=acc, loss_horizontal_confidence=loss_horizontal_confidence_log)
            print(lmbda)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss', losses.val, global_steps)
            writer.add_scalar('train_hkd_loss', loss_2d_log.val, global_steps)
            writer.add_scalar('train_vertical_loss', loss_vertical_log.val, global_steps)
            writer.add_scalar('train_horizontal_loss', loss_horizontal_log.val, global_steps)
            writer.add_scalar('train_horizontal_confidence_loss', loss_horizontal_confidence_log.val, global_steps)
            writer.add_scalar('train_mask_loss', loss_mask_log.val, global_steps)
            writer.add_scalar('train_acc', acc.val, global_steps)
            writer.add_scalar('degree_error', degree_error.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1
    return lmbda


def validate(config, val_loader, val_dataset, model, criterions,  output_dir,
             tb_log_dir, writer_dict=None, draw_pic=False, save_pickle=False):
    batch_time = AverageMeter()
    loss_hkd_log = AverageMeter()
    loss_vertical_log = AverageMeter()
    loss_horizontal_log = AverageMeter()
    loss_mask_log = AverageMeter()
    loss_iou_log = AverageMeter()
    losses = AverageMeter()
    degree_error = AverageMeter()
    acc = AverageMeter()
    Excellent = 0
    Mid_good = 0
    Poor_good = 0
    Poor_225 = 0
    Poor_45 = 0
    Total = 0

    # switch to evaluate mode
    model.eval()

    num_samples = len(val_dataset)
    idx = 0
    predictions = []
    confidences = []
    groundtruths = []
    with torch.no_grad():
        end = time.time()
        for i, (input, target, target_weight, degree, meta) in enumerate(val_loader):
            # compute output
            start = time.time()
            plane_output, hoe_output, conf = model(input)
            end = time.time()

            # change to cuda format
            target = target.cuda(non_blocking=True)
            target_weight = target_weight.cuda(non_blocking=True)
            degree = degree.cuda(non_blocking=True)
            loss_2d = criterions['2d_pose_loss'](plane_output, target, target_weight)
            loss_horizontal = criterions['hoe_loss'](hoe_output, degree)

            loss =  loss_horizontal
            num_images = input.size(0)
            mask = np.ones(num_images,dtype=bool)
            # measure accuracy and record loss
            # loss_hkd_log.update(loss_2d.item(), num_images)
            loss_horizontal_log.update(loss_horizontal.item(), num_images)
            losses.update(loss.item(), num_images)

            avg_degree_error, excellent, mid, poor_225 , poor, poor_45, _, gt_ori, pred_ori, cnt= comp_deg_error(hoe_output.detach().cpu().numpy()[mask],
                                                degree.detach().cpu().numpy()[mask])
            _, avg_acc, cnt, pred = accuracy(plane_output[:,:17].cpu().numpy(),
                                                target.cpu().numpy())
            acc.update(avg_acc, cnt)
            acc_label = 'kpd_acc'
            has_hkd = True

            max_hoe_output = hoe_output.detach().cpu().numpy().argmax(axis = 1)
            hoe_gt = degree.detach().cpu().numpy().argmax(axis = 1)

            if draw_pic:
                ori_path = os.path.join(output_dir, 'orientation_img')
                batch_keypoints = []
                for index in range(len(plane_output)):
                    keypoints = dict()
                    for j in range(23):
                        position = np.unravel_index(plane_output[index][j].cpu().numpy().argmax(), plane_output[index][j].cpu().numpy().shape)
                        keypoint = {"{}".format(j):np.array([position[0],position[1],plane_output[index][j].cpu().numpy().max()])}
                        keypoints.update(keypoint)
                    batch_keypoints.append(keypoints)
                if not os.path.exists(ori_path):
                    os.makedirs(ori_path)
                img_np = input.numpy()
                draw_conf_orientation(img_np, hoe_gt*5, max_hoe_output*5, ori_path, batch_keypoints, conf.cpu().numpy(), alis=str(i))

            if save_pickle:
                predictions.extend(list(hoe_output.detach().cpu().numpy()[mask]))
                confidences.extend(list(conf.detach().cpu().numpy()[mask]))
                # confidences.extend(list(horizontal_output.detach().cpu().numpy()))
                groundtruths.extend(np.array(list(gt_ori/5)))

            degree_error.update(avg_degree_error, num_images)

            Total += num_images
            # print(Total)
            Excellent += excellent
            Mid_good += mid
            Poor_good += poor
            Poor_45 += poor_45
            Poor_225 += poor_225

            # measure elapsed time
            batch_time.update(end - start)

            idx += num_images

            if i % config.PRINT_FREQ == 0:
                print_msg(step=i, loader_len=len(val_loader), batch_time=batch_time, has_hkd= has_hkd, loss_hkd=loss_hkd_log, loss_horizontal=loss_horizontal_log, loss_vertical=loss_vertical_log, loss_mask=loss_mask_log, losses=losses, degree_error=degree_error, acc_label=acc_label, acc=acc)

        if save_pickle:
            save_obj([predictions, confidences, groundtruths], 'vits_ori_conf_part_body')
        excel_rate = Excellent / Total
        mid_rate = Mid_good / Total
        poor_rate = Poor_good / Total
        poor_225_rate = Poor_225 / Total
        poor_45_rate = Poor_45 / Total
        name_values = {'Degree_error': degree_error.avg, '5_Excel_rate': excel_rate, '15_Mid_rate': mid_rate, '225_rate': poor_225_rate, '30_Poor_rate': poor_rate, '45_poor_rate': poor_45_rate}
        _print_name_value(name_values, config.MODEL.NAME)
        print(Total)
        if writer_dict:
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            writer.add_scalar(
                'valid_loss',
                losses.avg,
                global_steps
            )
            writer.add_scalar(
                'valid_hkd_loss',
                loss_hkd_log.avg,
                global_steps
            )
            writer.add_scalar(
                'valid_vertical_loss',
                loss_vertical_log.avg,
                global_steps
            )
            writer.add_scalar(
                'valid_horizontal_loss',
                loss_horizontal_log.avg,
                global_steps
            )
            writer.add_scalar(
                'valid_acc',
                acc.avg,
                global_steps
            )
            writer.add_scalar(
                'degree_error_val',
                degree_error.avg,
                global_steps
            )
            writer.add_scalar(
                'excel_rate',
                excel_rate,
                global_steps
            )
            writer.add_scalar(
                'mid_rate',
                mid_rate,
                global_steps
            )
            writer.add_scalar(
                'poor_rate',
                poor_rate,
                global_steps
            )

            writer_dict['valid_global_steps'] = global_steps + 1

        perf_indicator = degree_error.avg
    return perf_indicator

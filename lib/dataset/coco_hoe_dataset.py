from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import json
import cv2
import torch
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
import random
import numpy as np
from scipy.io import loadmat, savemat
from collections import OrderedDict
import torch.utils.data as data
from utils.copy_paste import CopyPaste
from utils.coco import CocoDetectionCP
from utils.visualize import display_instances
import albumentations as A


from pycocotools.coco import COCO
from lib.utils.transforms import get_affine_transform
from lib.utils.transforms import affine_transform
from lib.utils.transforms import fliplr_joints
from lib.utils.transforms import hoe_heatmap_gen, my_hoe_heatmap_gen
from lib.utils.utils import get_cos_similar_multi, hoe2horizon, hoe2vertical
from PIL import Image
logger = logging.getLogger(__name__)


class COCO_HOE_Dataset(data.Dataset):
    def __init__(self, cfg, root, is_train, transform=None):
        # self.renderer = getRenderer('cam')
        self.image_width = cfg.MODEL.IMAGE_SIZE[0]
        self.image_height = cfg.MODEL.IMAGE_SIZE[1]
        self.aspect_ratio = self.image_width * 1.0 / self.image_height
        self.pixel_std = 200
        self.root = root
        self.knee_joints_cnt = 0
        self.eft_counter = 0
        self.is_train = is_train
        if is_train:
            json_path = os.path.join(root, 'annotations', 'train_merged_eft_mebow.json')
            dataType = 'train_v1.0'
        else:
            json_path = os.path.join(root, 'annotations', 'val_merged_eft_mebow.json')
            dataType = 'val_v1.0'

        json_file = open(json_path, 'r')
        self.img_list = list(json.load(json_file).items())
        logger.info('=> load {} samples'.format(len(self.img_list)))
        print('=> load {} samples'.format(len(self.img_list)))

        annFile = os.path.join(root, 'annotations', 'coco_wholebody_{}.json'.format(dataType))
        self.coco_kps = COCO(annFile)

        # set parameters for key points
        self.num_joints = 23

        self.flip_pairs = [[1, 2], [3, 4], [5, 6], [7, 8],
                           [9, 10], [11, 12], [13, 14], [15, 16],[17,20],[18,21],[19,22]
                           ]

        self.upper_body_ids = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)

        self.lower_body_ids = (11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22)
        self.lower_body_ids_wo_ft = (11, 12, 13, 14, 15, 16)


        self.joints_weight = np.array(
            [
                1., 1., 1., 1., 1., 1., 1., 1.2, 1.2,
                1.2, 1.2, 1., 1., 1.2, 1.2, 1.5, 1.5, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0,
            ],
            dtype=np.float32
        ).reshape((self.num_joints, 1))

        self.use_different_joints_weight = cfg.LOSS.USE_DIFFERENT_JOINTS_WEIGHT

        # for data processing
        self.num_joints_half_body = cfg.DATASET.NUM_JOINTS_HALF_BODY
        self.prob_half_body = cfg.DATASET.PROB_HALF_BODY

        self.scale_factor = cfg.DATASET.SCALE_FACTOR
        self.rotation_factor = cfg.DATASET.ROT_FACTOR
        self.flip = cfg.DATASET.FLIP

        # generate heatmap label
        self.target_type = cfg.MODEL.TARGET_TYPE
        self.image_size = np.array(cfg.MODEL.IMAGE_SIZE)
        self.heatmap_size = np.array(cfg.MODEL.HEATMAP_SIZE)
        self.sigma = cfg.MODEL.SIGMA
        self.hoe_sigma = cfg.DATASET.HOE_SIGMA
        self.transform = transform

    def _box2cs(self, box):
        x, y, w, h = box[:4]
        return self._xywh2cs(x, y, w, h)

    def _xywh2cs(self, x, y, w, h):
        center = np.zeros((2), dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5

        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio
        scale = np.array(
            [w * 1.0 / self.pixel_std, h * 1.0 / self.pixel_std],
            dtype=np.float32)
        if center[0] != -1:
            scale = scale * 1.25
        return center, scale

    def half_body_transform(self, joints, joints_vis):
        lower_joints = []
        lower_joints_wo_foot = []
        for joint_id in range(self.num_joints):
            if joints_vis[joint_id][0] > 0:
                if joint_id in self.lower_body_ids:
                    lower_joints.append(joints[joint_id])
                if joint_id in self.lower_body_ids_wo_ft:
                    lower_joints_wo_foot.append(joints[joint_id])

        if len(lower_joints_wo_foot) > 2:
            selected_joints = lower_joints

            selected_joints = np.array(selected_joints, dtype=np.float32)

            left_top = np.amin(selected_joints, axis=0)
            right_bottom = np.amax(selected_joints, axis=0)
            center = np.array(lower_joints_wo_foot).mean(axis=0)[:2]
            # center = [(left_top[0] + right_bottom[0]) / 2.0, (left_top[1] + right_bottom[1]) / 2.0]
            # print(center)
            w = right_bottom[0] - left_top[0]
            h = right_bottom[1] - left_top[1]

            if w > self.aspect_ratio * h:
                h = w * 1.0 / self.aspect_ratio
            elif w < self.aspect_ratio * h:
                w = h * self.aspect_ratio

            scale = np.array(
                [
                    w * 1.0 / self.pixel_std,
                    h * 1.0 / self.pixel_std
                ],
                dtype=np.float32
            )
            scale = scale * 1.5
            return center, scale, True
        else:
            return None, None, False


    def __len__(self,):
        return len(self.img_list)

    def _load_image(self, index):
        str_id = self.img_list[index][0]
        img_id = int(str_id.split('_')[0])
        ann_id = int(str_id.split('_')[1])
        img_ann = self.coco_kps.loadImgs(img_id)[0]
        kps_ann = self.coco_kps.loadAnns(ann_id)[0]

        bbox = kps_ann['bbox']
        img_name = img_ann['file_name']
        if self.is_train:
            img_path = os.path.join(self.root, 'images', 'train2017', img_name)
        else:
            img_path = os.path.join(self.root, 'images', 'val2017', img_name)

        center, scale = self._box2cs(bbox)
        original_img_size = [img_ann['height'],img_ann['width']]
        eft_center = [0, 0]
        eft_scale = 0
        if(type(self.img_list[index][1]) == dict):
            degree = self.img_list[index][1]["orientation"]
        else:
            degree = self.img_list[index][1]
            # mask = None
        
        # label of orienation degree
        degree = int(degree) // 5
        
        if 'keypoints' not in kps_ann:
            logger.error('=> No keypoints')
            raise ValueError('No keypoints')
        kps_ann['keypoints'].extend(kps_ann["foot_kpts"])
        # kps_ann['keypoints'].extend([kps_ann["lefthand_kpts"][9],kps_ann["lefthand_kpts"][10],kps_ann["lefthand_kpts"][11], kps_ann["lefthand_kpts"][33],kps_ann["lefthand_kpts"][34],kps_ann["lefthand_kpts"][35], kps_ann["lefthand_kpts"][57], kps_ann["lefthand_kpts"][58], kps_ann["lefthand_kpts"][59]])
        # kps_ann['keypoints'].extend([kps_ann["righthand_kpts"][9],kps_ann["righthand_kpts"][10],kps_ann["righthand_kpts"][11], kps_ann["righthand_kpts"][33],kps_ann["righthand_kpts"][34],kps_ann["righthand_kpts"][35], kps_ann["righthand_kpts"][57], kps_ann["righthand_kpts"][58], kps_ann["righthand_kpts"][59]])
        joint = kps_ann['keypoints']
        if max(joint) == 0:
            logger.error('=> No joint {}'.format(joint))
            raise ValueError('No joint {}'.format(joint))
        joints_3d = np.zeros((self.num_joints, 3), dtype=np.float64)
        joints_3d_vis = np.zeros((self.num_joints, 3), dtype=np.float64)
        for ipt in range(self.num_joints):
            joints_3d[ipt, 0] = joint[ipt * 3 + 0]
            joints_3d[ipt, 1] = joint[ipt * 3 + 1]
            joints_3d[ipt, 2] = 0
            t_vis = joint[ipt * 3 + 2]
            if t_vis > 1:
                t_vis = 1
            if t_vis < 0:
                t_vis = 0
            joints_3d_vis[ipt, 0] = t_vis
            joints_3d_vis[ipt, 1] = t_vis
            joints_3d_vis[ipt, 2] = 0
        return img_path, center, scale, degree, joints_3d, joints_3d_vis, str_id, eft_center, eft_scale

    def __getitem__(self, index):
        lower_body = False
        imgfile, c, s, degree, joints, joints_vis, str_id, eft_center, eft_scale = self._load_image(index)
        data_numpy = cv2.imread(imgfile, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)      
        data_numpy = cv2.cvtColor(data_numpy, cv2.COLOR_BGR2RGB)

        original_image_size = data_numpy.shape[:2]

        if data_numpy is None:
            logger.error('=> fail to read {}'.format(imgfile))
            raise ValueError('Fail to read {}'.format(imgfile))

        # Not use score
        # score = 0

        if (np.sum(joints_vis[:17, 0]) > 8):
            c_half_body, s_half_body, lower_body = self.half_body_transform(
                joints, joints_vis
            )
            if c_half_body is not None and s_half_body is not None:
                c, s = c_half_body, s_half_body

        degree = hoe_heatmap_gen(degree, 72, sigma=self.hoe_sigma)
        trans = get_affine_transform(c, s, 0, self.image_size)
        input = cv2.warpAffine(
            data_numpy,
            trans,
            (int(self.image_size[0]), int(self.image_size[1])),
            flags=cv2.INTER_LINEAR)

        for i in range(self.num_joints):
            if joints_vis[i, 0] > 0.0:
                joints[i, 0:2] = affine_transform(joints[i, 0:2], trans)
        target, target_weight = self.generate_target(joints, joints_vis)

        if self.transform:
            input = self.transform(input)
        input = input.float()

        meta = {
            'image_path': imgfile,
            'center': c,
            'scale': s,
            'lower_body': lower_body,
        }
        return input, target, target_weight, degree, meta


    def generate_target(self, joints, joints_vis):
        '''
        :param joints:  [num_joints, 3]
        :param joints_vis: [num_joints, 3]
        :return: target, target_weight(1: visible, 0: invisible)
        '''
        target_weight = np.ones((self.num_joints, 1), dtype=np.float32)
        target_weight[:, 0] = joints_vis[:, 0]

        assert self.target_type == 'gaussian', \
            'Only support gaussian map now!'

        if self.target_type == 'gaussian':
            target = np.zeros((self.num_joints,
                               self.heatmap_size[1],
                               self.heatmap_size[0]),
                              dtype=np.float32)

            tmp_size = self.sigma * 3

            for joint_id in range(self.num_joints):
                feat_stride = self.image_size / self.heatmap_size
                mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)
                mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)
                # Check that any part of the gaussian is in-bounds
                ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
                br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
                if ul[0] >= self.heatmap_size[0] or ul[1] >= self.heatmap_size[1] \
                        or br[0] < 0 or br[1] < 0:
                    # If not, just return the image as is
                    target_weight[joint_id] = 0
                    continue

                # # Generate gaussian
                size = 2 * tmp_size + 1
                x = np.arange(0, size, 1, np.float32)
                y = x[:, np.newaxis]
                x0 = y0 = size // 2
                # The gaussian is not normalized, we want the center value to equal 1
                g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.sigma ** 2))

                # Usable gaussian range
                g_x = max(0, -ul[0]), min(br[0], self.heatmap_size[0]) - ul[0]
                g_y = max(0, -ul[1]), min(br[1], self.heatmap_size[1]) - ul[1]
                # Image range
                img_x = max(0, ul[0]), min(br[0], self.heatmap_size[0])
                img_y = max(0, ul[1]), min(br[1], self.heatmap_size[1])

                v = target_weight[joint_id]
                if v > 0.5:
                    target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                        g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
        if self.use_different_joints_weight:
            target_weight = np.multiply(target_weight, self.joints_weight)

        return target, target_weight
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import time
from collections import namedtuple
from pathlib import Path
import numpy as np
import cv2

import torch
import torch.optim as optim
import torch.nn as nn

def hoe2vertical(label):
    # 0~36
    if label <= 36:
        return label
    else:
        return 72-label

def draw_dense_reg(hm_size, joint, sigma, locref_stdev, kpd):

    offset_map = np.empty([2, hm_size[1], hm_size[0]], dtype=np.float32)  # [2, height, width]

    y, x = np.ogrid[0:hm_size[1], 0:hm_size[0]]
    mat_x, mat_y = np.meshgrid(x, y)  # standard cooridate field

    offset_map[0] = joint[0] - mat_x  # x-axis offset field
    offset_map[1] = joint[1] - mat_y  # y-axis offset field

    h = np.sum(offset_map*offset_map, axis=0)  # distance**2, [height, width]
    heatmap = np.exp(- h / (2 * sigma ** 2))
    heatmap[heatmap < np.finfo(heatmap.dtype).eps * heatmap.max()] = 0  # gaussian heatmap

    offset_map /= locref_stdev  # rescale offset map

    mask01 = np.where(h <= kpd**2, 1, 0)  # 0-1 mask
    offset_map *= mask01[None, ...]
        
    return heatmap, offset_map

def hoe2horizon(label):
    # 37
    up = [18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0,71,70,69,68,67,66,65,64,63,62,61,60,59,58,57,56,55,54]
    # 37
    down = [18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54]
    if label in down:
        return int(down.index(label))
    else:
        return int(up.index(label))

def vh2hoe(verticals, horizontals):
    up =   [18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0,71,70,69,68,67,66,65,64,63,62,61,60,59,58,57,56,55,54]
    down = [18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54]
    results = []
    for vertical, horizontal in zip(verticals, horizontals):
        if vertical <= 18:
            results.append(up[horizontal])
        else:
            results.append(down[horizontal])
    return np.array(results)

def bin2hoe(verticals, horizontals):
    up =   [18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0,71,70,69,68,67,66,65,64,63,62,61,60,59,58,57,56,55,54]
    down = [18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54]
    results = []
    for vertical, horizontal in zip(verticals, horizontals):
        # print(vertical.argmax())
        if vertical.argmax() == 0:
            results.append(up[horizontal])
        else:
            results.append(down[horizontal])
    return np.array(results)

# def bin2hoe(verticals, horizontals):
#     up =   [18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0,71,70,69,68,67,66,65,64,63,62,61,60,59,58,57,56,55,54]
#     down = [18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54]
#     results = []
#     for vertical, horizontal in zip(verticals, horizontals):
#         if vertical <= 0.5:
#             results.append(down[horizontal])
#         else:
#             results.append(up[horizontal])
#     return np.array(results)

def get_key_by_value(d, value):
    for k, v in d.items():
        if v == value:
            return k
    raise ValueError('Value not found in dictionary.')


def get_cos_similar_multi(v1: list, v2: list):
    num = np.dot([v1], np.array(v2).T)  
    denom = np.linalg.norm(v1) * np.linalg.norm(v2, axis=1) 
    res = num / denom
    res[np.isneginf(res)] = 0
    return res

def create_logger(cfg, cfg_name, phase='train'):
    root_output_dir = Path(cfg.OUTPUT_DIR)
    # set up logger
    if not root_output_dir.exists():
        print('=> creating {}'.format(root_output_dir))
        root_output_dir.mkdir()

    dataset = cfg.DATASET.DATASET + '_' + cfg.DATASET.HYBRID_JOINTS_TYPE \
        if cfg.DATASET.HYBRID_JOINTS_TYPE else cfg.DATASET.DATASET
    dataset = dataset.replace(':', '_')
    model = cfg.MODEL.NAME
    cfg_name = os.path.basename(cfg_name).split('.')[0]

    final_output_dir = root_output_dir / dataset / model / cfg_name

    print('=> creating {}'.format(final_output_dir))
    final_output_dir.mkdir(parents=True, exist_ok=True)

    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}_{}.log'.format(cfg_name, time_str, phase)
    final_log_file = final_output_dir / log_file
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    tensorboard_log_dir = Path(cfg.LOG_DIR) / dataset / model / \
        (cfg_name + '_' + time_str)

    print('=> creating {}'.format(tensorboard_log_dir))
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)

    return logger, str(final_output_dir), str(tensorboard_log_dir)

def restrain_upper_bound(x):
    if x >= 224:
        x = 224
    elif x <= 0:
        x = 0
    else:
        x = x
    return x

def joints_to_mask(new_selected_joints):
    if len(new_selected_joints) == 0:
        return  np.zeros((64, 48))
    new_selected_joints = list(map(restrain_upper_joints_bound, new_selected_joints))
    left_top = np.amin(new_selected_joints, axis=0)
    right_bottom = np.amax(new_selected_joints, axis=0)
    bbox_mask = np.zeros((64,48, 3))
    bbox_mask = cv2.rectangle(bbox_mask, (int(left_top[0]/4), int(left_top[1]/4)), (int(right_bottom[0]/4), int(right_bottom[1]/4)), [255,255,255], -1)
    bbox_mask = bbox_mask.transpose(2,0,1)[0]

    return bbox_mask

def restrain_upper_joints_bound(x):
    if x[0] <= 0:
        x[0] = 0
    elif x[0] >= 192:
        x[0] = 192
    else:
        x[0] = x[0]

    if x[1] <= 0:
        x[1] = 0
    elif x[1] >= 256:
        x[1] = 256
    else:
        x[1] = x[1]
    return x


def backprojected_mask(crop_image, center, scale, original_size):
    crop_width = 200 * scale
    crop_height = 200 * scale

    crop_image = crop_image.transpose(1,2,0)
    img_resized = cv2.resize(crop_image, (int(crop_width), int(crop_height)))
    img_backprojected = np.zeros((original_size[0], original_size[1], 24), np.uint8)

    x_start = int(center[1] - crop_width / 2)
    y_start = int(center[0] - crop_width / 2)
    x_end = x_start + int(crop_width)
    y_end = y_start + int(crop_height)

    x_project_start = x_start
    x_project_end = x_end
    y_project_start = y_start
    y_project_end = y_end

    x_resize_start = 0
    x_resize_end = -1
    y_resize_start = 0
    y_resize_end = -1
    if x_start < 0:
        x_project_start = 0
        x_resize_start = 0-x_start
    else:
        x_project_start = x_start
        x_resize_start = 0

    if x_end > original_size[0]:
        x_project_end = original_size[0]
        x_resize_end = original_size[0] - x_end
    else:
        x_project_end = x_end
        x_resize_end = int(crop_height)

    if y_start < 0:
        y_project_start = 0
        y_resize_start = 0-y_start
    else:
        y_project_start = y_start
        y_resize_start = 0


    if y_end > original_size[1]:
        y_project_end = original_size[1]
        y_resize_end = original_size[1] - y_end
    else:
        y_project_end = y_end
        y_resize_end = int(crop_width)


    img_backprojected[x_project_start:x_project_end, y_project_start:y_project_end, :] = img_resized[x_resize_start:x_resize_end,y_resize_start:y_resize_end,:]

    img_backprojected = img_backprojected.transpose(2,0,1)
    return img_backprojected


def get_optimizer(cfg, model):
    optimizer = None
    if cfg.TRAIN.OPTIMIZER == 'sgd':
        optimizer = optim.SGD(
            model.get_hoe_params(),
            lr=cfg.TRAIN.LR,
            momentum=cfg.TRAIN.MOMENTUM,
            weight_decay=cfg.TRAIN.WD,
            nesterov=cfg.TRAIN.NESTEROV
        )
    elif cfg.TRAIN.OPTIMIZER == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            # model.module.get_hoe_params(),
            lr=cfg.TRAIN.LR
        )
    elif cfg.TRAIN.OPTIMIZER == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=cfg.TRAIN.LR,
            eps=1e-8,
            betas=[0.9,0.999],
            weight_decay=0.05
        )


    return optimizer


def save_checkpoint(states, is_best, output_dir,
                    filename='checkpoint.pth'):
    torch.save(states, os.path.join(output_dir, filename))
    if is_best and 'state_dict' in states:
        torch.save(states['best_state_dict'],
                    os.path.join(output_dir, 'model_best.pth'))


def get_model_summary(model, *input_tensors, item_length=26, verbose=False):
    """
    :param model:
    :param input_tensors:
    :param item_length:
    :return:
    """

    summary = []

    ModuleDetails = namedtuple(
        "Layer", ["name", "input_size", "output_size", "num_parameters", "multiply_adds"])
    hooks = []
    layer_instances = {}

    def add_hooks(module):

        def hook(module, input, output):
            class_name = str(module.__class__.__name__)

            instance_index = 1
            if class_name not in layer_instances:
                layer_instances[class_name] = instance_index
            else:
                instance_index = layer_instances[class_name] + 1
                layer_instances[class_name] = instance_index

            layer_name = class_name + "_" + str(instance_index)

            params = 0

            if class_name.find("Conv") != -1 or class_name.find("BatchNorm") != -1 or \
               class_name.find("Linear") != -1:
                for param_ in module.parameters():
                    params += param_.view(-1).size(0)

            flops = "Not Available"
            if class_name.find("Conv") != -1 and hasattr(module, "weight"):
                flops = (
                    torch.prod(
                        torch.LongTensor(list(module.weight.data.size()))) *
                    torch.prod(
                        torch.LongTensor(list(output.size())[2:]))).item()
            elif isinstance(module, nn.Linear):
                flops = (torch.prod(torch.LongTensor(list(output.size()))) \
                         * input[0].size(1)).item()

            if isinstance(input[0], list):
                input = input[0]
            if isinstance(output, list):
                output = output[0]

            summary.append(
                ModuleDetails(
                    name=layer_name,
                    input_size=list(input[0].size()),
                    output_size=list(output[0].size()),
                    num_parameters=params,
                    multiply_adds=flops)
            )

        if not isinstance(module, nn.ModuleList) \
           and not isinstance(module, nn.Sequential) \
           and module != model:
            hooks.append(module.register_forward_hook(hook))

    model.eval()
    model.apply(add_hooks)

    space_len = item_length

    model(*input_tensors)
    for hook in hooks:
        hook.remove()

    details = ''
    if verbose:
        details = "Model Summary" + \
            os.linesep + \
            "Name{}Input Size{}Output Size{}Parameters{}Multiply Adds (Flops){}".format(
                ' ' * (space_len - len("Name")),
                ' ' * (space_len - len("Input Size")),
                ' ' * (space_len - len("Output Size")),
                ' ' * (space_len - len("Parameters")),
                ' ' * (space_len - len("Multiply Adds (Flops)"))) \
                + os.linesep + '-' * space_len * 5 + os.linesep

    params_sum = 0
    flops_sum = 0
    for layer in summary:
        params_sum += layer.num_parameters
        if layer.multiply_adds != "Not Available":
            flops_sum += layer.multiply_adds
        if verbose:
            details += "{}{}{}{}{}{}{}{}{}{}".format(
                layer.name,
                ' ' * (space_len - len(layer.name)),
                layer.input_size,
                ' ' * (space_len - len(str(layer.input_size))),
                layer.output_size,
                ' ' * (space_len - len(str(layer.output_size))),
                layer.num_parameters,
                ' ' * (space_len - len(str(layer.num_parameters))),
                layer.multiply_adds,
                ' ' * (space_len - len(str(layer.multiply_adds)))) \
                + os.linesep + '-' * space_len * 5 + os.linesep

    details += os.linesep \
        + "Total Parameters: {:,}".format(params_sum) \
        + os.linesep + '-' * space_len * 5 + os.linesep
    details += "Total Multiply Adds (For Convolution and Linear Layers only): {:,} GFLOPs".format(flops_sum/(1024**3)) \
        + os.linesep + '-' * space_len * 5 + os.linesep
    details += "Number of Layers" + os.linesep
    for layer in layer_instances:
        details += "{} : {} layers   ".format(layer, layer_instances[layer])

    return details

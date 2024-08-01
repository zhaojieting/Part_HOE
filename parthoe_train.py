# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Chenyan Wu (czw390@psu.edu)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

import pprint
import shutil
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

import _init_paths
from config import cfg
from config import update_config

from core.loss import JointsMSELoss
from core.loss import HoeMSELoss

from core.function import train
from core.function import validate

from utils.utils import get_optimizer
from utils.utils import save_checkpoint
from utils.utils import create_logger
from utils.utils import get_model_summary

import dataset
from models import ViT


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    # philly
    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--prevModelDir',
                        help='prev Model directory',
                        type=str,
                        default='')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    update_config(cfg, args)

    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, args.cfg, 'train')

    logger.info(pprint.pformat(args))
    logger.info(cfg)
    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    model = ViT(
        img_size=(256,192), 
        patch_size=16, 
        in_chans=3, 
        embed_dim=384, 
        depth=12,
        num_heads=12, 
        ratio=1, 
        use_checkpoint=False,
        mlp_ratio=4,
        qkv_bias=True, 
        drop_path_rate=0.1,
        )

    # joints pretrain
    # checkpoint = torch.load("/home/zhaojieting/jieting_ws/MEBOW/models/vit+s-coco.pth", map_location='cpu')
    
    # mae pretrain
    checkpoint = torch.load("/home/zhaojieting/jieting_ws/MEBOW/models/small_pretrained.pth", map_location='cpu')

    # import pdb;pdb.set_trace()
    for name, param in model.named_parameters():
        try:
            if name == 'decoder.conv_for_heatmap1.weight':
                param.data = checkpoint['state_dict']['keypoint_head.deconv_layers.0.weight']
            elif name == 'decoder.conv_for_heatmap2.weight':
                param.data = checkpoint['state_dict']['keypoint_head.deconv_layers.3.weight']
            elif name == 'decoder.bn1.weight':
                param.data = checkpoint['state_dict']['keypoint_head.deconv_layers.1.weight']
            elif name == 'decoder.bn2.bias':
                param.data = checkpoint['state_dict']['keypoint_head.deconv_layers.1.bias']
            elif name == 'decoder.bn2.weight':
                param.data = checkpoint['state_dict']['keypoint_head.deconv_layers.4.weight']
            elif name == 'decoder.bn1.bias':
                param.data = checkpoint['state_dict']['keypoint_head.deconv_layers.4.bias']
            # elif name == "decoder.heatmap_final.weight":
            #     param.data = checkpoint['state_dict']['keypoint_head.final_layer.weight']
            # elif name == "decoder.heatmap_final.bias":
            #     param.data = checkpoint['state_dict']['keypoint_head.final_layer.bias']                
            elif name == "pos_embed":
                param.data = checkpoint['model'][name].data[:,:193,:]
            else:
                # param.data = checkpoint['state_dict']['backbone.'+name].data

                # 记得这里需要更改名字,当改变预训练文件的时候
                param.data = checkpoint['model'][name].data

        except Exception as e:
            print(f'Failed to load parameter{name}: {e}')
    # copy model file
    this_dir = os.path.dirname(__file__)
    # shutil.copy2(
    #     os.path.join(this_dir, '../lib/models', cfg.MODEL.NAME + '.py'),
    #     final_output_dir)
    # logger.info(pprint.pformat(model))

    writer_dict = {
        'writer': SummaryWriter(logdir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    dump_input = torch.rand(
        (1, 3, cfg.MODEL.IMAGE_SIZE[1], cfg.MODEL.IMAGE_SIZE[0])
    )
    # writer_dict['writer'].add_graph(model, (dump_input, ), operator_export_type = "RAW")

    logger.info(get_model_summary(model, dump_input))

    # load pretrained model

    # pre_model = '/home/cchw/coding/important_model/amazon_model/gray_HM3.6_MPII_with_HOE.pth'
    # logger.info("=> loading checkpoint '{}'".format(pre_model))
    # checkpoint = torch.load(pre_model)
    # if 'state_dict' in checkpoint:
    #     model.load_state_dict(checkpoint['state_dict'], strict=True)
    # else:
    #     model.load_state_dict(checkpoint, strict=True)

    # import pdb;pdb.set_trace()
    model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()


    criterions = {}
    criterions['2d_pose_loss'] = JointsMSELoss(
        use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT
    ).cuda()
    # criterions['hoe_loss'] = torch.nn.NLLLoss().cuda()
    criterions['vertical_loss'] = torch.nn.MSELoss().cuda()
    # criterions['vertical_loss'] = torch.nn.CrossEntropyLoss().cuda()
    # criterions['hoe_loss'] = SoftTargetCrossEntropy().cuda()
    criterions['hoe_loss'] = HoeMSELoss().cuda()
    criterions['vh_constraint'] = VhConstraintLoss().cuda()
    # criterions['mask_loss'] = MasksCrossEntropy().cuda()
    criterions['mask_loss'] = DiceLoss(weight=torch.tensor([0.4, 0.6, 0.4, 0.4, 0.6, 0.6, 0.4, 0.6, 0.6])).cuda()
    criterions['cross_mask_loss'] = MasksCrossEntropy(weight=torch.tensor([0.01, 0.4, 0.2, 0.4, 0.8, 0.8, 0.4, 0.8, 0.8])).cuda()
    # criterions['mask_loss'] = MasksCrossEntropy(weight=torch.tensor([0.1, 0.5, 0.2, 0.2, 0.2])).cuda()
    # criterions['mask_loss'] = MasksCrossEntropy(weight=torch.tensor([0.1, 0.8, 0.2, 0.5, 0.3, 1.0,0.5, 0.3, 1.0])).cuda()
    # criterions['fea_loss'] = HwMasksCrossEntropy().cuda()
    # criterions['mask_loss'] = BodyPartAttentionLoss(weight=torch.tensor([0.1, 0.8, 0.2, 0.5, 0.5, 0.3, 0.3, 1.0, 1.0])).cuda()
    criterions['spatial_consistency_loss'] = SpatialConsistencyLoss().cuda()
    criterions['weight_loss'] = Weight_loss().cuda()
    criterions['iou_loss'] = MaskIOU_loss().cuda()
    criterions['contrast_loss'] = SupConLoss().cuda()

    # Data loading code
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    # normalize = transforms.Normalize(
    #     mean=[0.3281186, 0.28937867, 0.20702125], std=[0.09407319, 0.09732835, 0.106712654]
    # )

    train_dataset = eval('dataset.'+cfg.DATASET.DATASET)(
        cfg, cfg.DATASET.TRAIN_ROOT, True,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )


    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU*len(cfg.GPUS),
        shuffle=cfg.TRAIN.SHUFFLE,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )

    # import pdb;pdb.set_trace()
    valid_dataset = eval('dataset.' + cfg.DATASET.DATASET)(
        cfg, cfg.DATASET.VAL_ROOT, False,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )


    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU*len(cfg.GPUS),
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )

    best_perf = 200.0
    best_model = False
    last_epoch = -1
    optimizer = get_optimizer(cfg, model)
    begin_epoch = cfg.TRAIN.BEGIN_EPOCH

    checkpoint_file = os.path.join(
        final_output_dir, 'checkpoint.pth'
    )

    if cfg.AUTO_RESUME and os.path.exists(checkpoint_file):
        logger.info("=> loading checkpoint '{}'".format(checkpoint_file))
        checkpoint = torch.load(checkpoint_file)
        # import pdb;pdb.set_trace()
        begin_epoch = checkpoint['epoch']
        best_perf = checkpoint['perf']
        last_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])

        optimizer.load_state_dict(checkpoint['optimizer'])
        logger.info("=> loaded checkpoint '{}' (epoch {})".format(
            checkpoint_file, checkpoint['epoch']))

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, cfg.TRAIN.LR_STEP, cfg.TRAIN.LR_FACTOR,
        last_epoch=last_epoch
    )
    lmbda = 0.00001
    # best_perf = 20
    for epoch in range(begin_epoch, cfg.TRAIN.END_EPOCH):
        lmbda = bin_train(cfg, train_loader, train_dataset, model, criterions, optimizer, epoch,
              final_output_dir, tb_log_dir, writer_dict, lmbda)

        perf_indicator = bin_validate(
            cfg, valid_loader, valid_dataset, model, criterions,
            final_output_dir, tb_log_dir, writer_dict, draw_pic=False, save_pickle=False
        )
        if perf_indicator <= best_perf:
            best_perf = perf_indicator
            best_model = True
        else:
            best_model = False
        current_lr = optimizer.param_groups[0]['lr']
        lr_scheduler.step()
        logger.info('=> saving checkpoint to {}'.format(final_output_dir))
        logger.info('best_model{}'.format(best_perf))
        save_checkpoint({
            'epoch': epoch + 1,
            'model': cfg.MODEL.NAME,
            'state_dict': model.state_dict(),
            'best_state_dict': model.module.state_dict(),
            'perf': best_perf,
            'optimizer': optimizer.state_dict(),
        }, best_model, final_output_dir)

    final_model_state_file = os.path.join(
        final_output_dir, 'final_checkpoint.pth'
    )
    logger.info('=> saving final model state to {}'.format(
        final_model_state_file)
    )
    torch.save(model.module.state_dict(), final_model_state_file)
    writer_dict['writer'].close()


if __name__ == '__main__':
    main()

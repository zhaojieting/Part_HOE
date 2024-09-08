# Part_HOE
This is the official implementation of our article: "Human Orientation Estimation Under Partial Observation".
<img src="https://github.com/zhaojieting/Part_HOE/blob/main/docs/IROS2024-Video-2x.gif" width="760" height="480" />

## Dataset Preparation
..

## Pretrained checkpoints
Download the pretrained parthoe_s checkpoint, and put it in the checkpoints folder.
https://drive.google.com/file/d/1M4Jr2IQ8p8PQjXPWVcAuSHwGMVh6hdX6/view?usp=drive_link

## Test
python parthoe_test.py --cfg config/parthoe.yaml

## Train
python parthoe_train.py --cfg config/parthoe.yaml

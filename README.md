# Part_HOE
This is the official implementation of our article: "Human Orientation Estimation Under Partial Observation".
<img src="https://github.com/zhaojieting/Part_HOE/blob/main/docs/IROS2024-Video-2x.gif" width="760" height="480" />

## Environment Setup
```
conda create -n part_hoe python=3.7
conda activate part_hoe
pip install -r requirement.txt
pip install timm==0.4.9 einops
```
## Dataset Preparation
   ```
   # COCOAPI=/path/to/clone/cocoapi
   git clone https://github.com/cocodataset/cocoapi.git $COCOAPI
   cd $COCOAPI/PythonAPI
   # Install into global site-packages
   make install
   # Alternatively, if you do not have permissions or prefer
   # not to install the COCO API into global site-packages
   python3 setup.py install --user
   ```

## Pretrained checkpoints
Download the pre-trained parthoe_s checkpoint(https://drive.google.com/file/d/1M4Jr2IQ8p8PQjXPWVcAuSHwGMVh6hdX6/view?usp=drive_link
), and put it in the checkpoints folder.

## Test
python parthoe_test.py --cfg config/parthoe.yaml

## Train
If you want to retrain the model, you must download the pre-trained backbone checkpoint here(https://drive.google.com/file/d/1M4Jr2IQ8p8PQjXPWVcAuSHwGMVh6hdX6/view?usp=drive_link).
python parthoe_train.py --cfg config/parthoe.yaml

## Acknowledge
Since I graduated from SUSTech in 2024.07, the code and dataset might have some bugs ~ 
If you need any help, feel free to ask me by GitHub issues!
This work is inspired from MEBOW(), ViTPose()

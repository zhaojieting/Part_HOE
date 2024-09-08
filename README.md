# [Part_HOE](https://arxiv.org/abs/2404.14139)
Since I graduated from school in 2024.07, the environment has not remained the original version, and the code might have some bugs ~  

If you need any help, **feel free to ask me through GitHub issues**!
<img src="https://github.com/zhaojieting/Part_HOE/blob/main/docs/IROS2024-Video-2x.gif" width="760" height="480" />

## Environment Setup
   ```
   conda create -n part_hoe python=3.7
   conda activate part_hoe
   pip install -r requirement.txt
   pip install timm==0.4.9 einops
   ```
## Dataset Preparation
### Install COCOAPI
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

### Prepare the Dataset, whole-body joint annotations, and orientation annotations 
Download 4 [annotations](https://drive.google.com/drive/folders/1J3xDMaJMF25nTjO7li9d-UKh8_16zPHf?usp=drive_link) and put them into the COCO annotation folder.
```
|-- data
`-- |-- coco
    `-- |-- annotations
        |   |-- coco_wholebody_train_v1.0.json
        |   |-- coco_wholebody_val_v1.0.json
        |   |-- merged_orientation_train.json
        |   |-- merged_orientation_val.json
        `-- images
            |-- train2017
            |   |-- 000000000009.jpg
            |   |-- 000000000025.jpg
            |   |-- 000000000030.jpg
            |   |-- ... 
            `-- val2017
                |-- 000000000139.jpg
                |-- 000000000285.jpg
                |-- 000000000632.jpg
                |-- ...
```
## Test
Download the [PartHOE_S](https://drive.google.com/file/d/1M4Jr2IQ8p8PQjXPWVcAuSHwGMVh6hdX6/view?usp=drive_link
) weight and put it in the checkpoints folder.
```
python parthoe_test.py --cfg config/parthoe.yaml
```
## Train
If you want to retrain the model, you must download the pre-trained [vit-s](https://drive.google.com/file/d/1M4Jr2IQ8p8PQjXPWVcAuSHwGMVh6hdX6/view?usp=drive_link) weight.
```
python parthoe_train.py --cfg config/parthoe.yaml
```
## Acknowledgement
The code of this work is based on the open-source project [MEBOW](https://github.com/ChenyanWu/MEBOW) and [ViTPose](https://github.com/ViTAE-Transformer/ViTPose/tree/main).

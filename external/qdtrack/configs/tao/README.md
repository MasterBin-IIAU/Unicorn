## Experiments on TAO Dataset


### 1. Download the data
a. Please follow [TAO download](https://github.com/TAO-Dataset/tao/blob/master/docs/download.md) instructions.

b. Please also prepare the [LVIS dataset](https://www.lvisdataset.org/).

It is recommended to symlink the dataset root to `$QDTrack/data`.

If your folder structure is different, you may need to change the corresponding paths in config files.

Our folder structure follows

```
├── qdtrack
├── tools
├── configs
├── data
    ├── tao
        ├── frames
            ├── train
            ├── val
            ├── test
        ├── annotations
    ├── lvis
        ├── train2017
        ├── annotations    
```

### 2. Install the TAO API

```shell
pip install git+https://github.com/OceanPang/tao.git
```

We fork the TAO API to make the logger compatabile with our codebase. 

We also print the AP of main classes such as "person" for reference. 

### 3. Generate our annotation files

a. Generate TAO annotation files with 482 classes.
```shell
python tools/convert_datasets/tao2coco.py -t ./data/tao/annotations --filter-classes
```

b. Merge LVIS and COCO training sets.

Use the `merge_coco_with_lvis.py` script in [the offical TAO API](https://github.com/TAO-Dataset/tao/blob/master/scripts/detectors/merge_coco_with_lvis.py).

This operation follows the paper [TAO](https://taodataset.org/).

```shell
cd ${TAP_API}
python ./scripts/detectors/merge_coco_with_lvis.py --lvis ${LVIS_PATH}/annotations/lvis_v0.5_train.json --coco ${COCO_PATH}/annotations/instances_train2017.json --mapping data/coco_to_lvis_synset.json --output-json ${LVIS_PATH}/annotations/lvisv0.5+coco_train.json
```

You can also get the merged annotation file from [Google Drive](https://drive.google.com/file/d/1v_q0eWpKgVDMvmjQ8pBKPgHQQ8SLhLx0/view?usp=sharing) or [Baidu Yun](https://pan.baidu.com/s/1XnwJ5FqsA_neV0MSXu42hg) (passcode: rkh2).

### 4. Pre-train the model on LVIS dataset

a. Pre-train our QDTrack on LVISv0.5+COCO2017 training set.

```shell
sh ./tools/dist_train.sh ./configs/tao/qdtrack_frcnn_r101_fpn_24e_lvis.py 8
```

The detection performance on TAO validation is

| AP | AP50 | AP75 | AP_S | AP_M | AP_L |
|----|------|------|------|------|------|
|17.3| 29.2 | 17.4 | 5.7  | 13.1 | 22.1 |

Here is a checkpoint at [Google Drive](https://drive.google.com/file/d/1XYe4BiYbBQIGMSuI5Ht6KXtuCEI4MTY4/view?usp=sharing). and [Baidu Yun](https://pan.baidu.com/s/1P1vMNeHHNgrl0kdV9upfjg) (passcode: i4rm).

b. Save the runned model to `ckpts/tao/**.pth`, and modify the configs for TAO accordingly.


### 5. Fine-tune the model on TAO dataset

```shell
sh ./tools/dist_train.sh ./configs/tao/qdtrack_frcnn_r101_fpn_12e_tao_ft.py 8
```
You can found a trained model at [Google Drive](https://drive.google.com/file/d/1JtZ9UA0-b9LDor1NHtk8A-g83X7-T89X/view?usp=sharing) and [Baidu Yun](https://pan.baidu.com/s/1DxQNu_JgkEpwfjPPVStcEg) (passcode: uhjq).

Results on TAO validation set:

| AP(50:75) | AP50 | AP75 | AP_S(50:75) | AP_M (50:75)| AP_L(50:75) |
|----|------|------|------|------|------|
| 11.2 | 15.9 | 6.4 | 7.9  | 9.2 | 14.7 |
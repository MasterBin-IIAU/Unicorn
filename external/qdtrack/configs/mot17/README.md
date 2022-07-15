## Experiments on MOT17 Dataset

### 1. Download and prepare the data

Please download the data from MOT Challenge.

It is recommended to symlink the dataset root to `$QDTrack/data`.

If your folder structure is different, you may need to change the corresponding paths in config files.

Our folder structure follows

```
├── qdtrack
├── tools
├── configs
├── data
    ├── MOT17
        ├── train
        ├── test
```


### 2. Generate our annotation files

```shell
python tools/convert_datasets/mot2coco.py -i ./data/MOT17 -o data/MOT17/annotations --split-train --convert-det
```

### 3. Train the model on MOT17

```shell
sh ./tools/dist_train.sh ./configs/mot17/qdtrack_frcnn_r50_fpn_4e_mot17.py 8
```

The pretrained model from MSCOCO can be obtained from [Google Drive](https://drive.google.com/file/d/1xK4Gvtd_2OchAyRY5WcoMkqoE9UM4gFO/view?usp=sharing) and [Baidu Yun](https://pan.baidu.com/s/1F9rLqsHjOv9DyJ-guvM-Qw) (passcode: vcu1).


# Data Preparation

We put pretrained backbone weights under ${UNICORN_ROOT} and put all data under the `datasets` folder. The complete data structure looks like this.
```
${UNICORN_ROOT}
    -- convnext_tiny_1k_224_ema.pth
    -- convnext_large_22k_224.pth
    -- datasets
        -- bdd
            -- images
                -- 10k
                -- 100k
                -- seg_track_20
                -- track
            -- labels
                -- box_track_20
                -- det_20
                -- ins_seg
                -- seg_track_20
        -- Cityscapes
            -- annotations
            -- images
            -- labels_with_ids
        -- COCO
            -- annotations
            -- train2017
            -- val2017
        -- crowdhuman
            -- annotations
            -- CrowdHuman_train
            -- CrowdHuman_val
            -- annotation_train.odgt
            -- annotation_val.odgt
        -- DAVIS
            -- Annotations
            -- ImageSets
            -- JPEGImages
            -- README.md
            -- SOURCES.md
        -- ETHZ
            -- annotations
            -- eth01
            -- eth02
            -- eth03
            -- eth05
            -- eth07
        -- GOT10K
            -- test
                -- GOT-10k_Test_000001
                -- ...
            -- train
                -- GOT-10k_Train_000001
                -- ...
        -- LaSOT
            -- airplane
            -- basketball
            -- ...
        -- mot
            -- annotations
            -- test
            -- train
        -- MOTS
            -- annotations
            -- test
            -- train
        -- saliency
            -- image
            -- mask
        -- TrackingNet
            -- TEST
            -- TRAIN_0
            -- TRAIN_1
            -- TRAIN_2
            -- TRAIN_3
        -- ytbvos18
            -- train
            -- val
```


## Pretrained backbone weights
Unicorn uses [ConvNeXt](https://arxiv.org/abs/2201.03545) as the backbone by default. The pretrained backbone weights can be downloaded by the following commands.
```
wget -c https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth # convnext-tiny
wget -c https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_224.pth # convnext-large
```


## Data
For users who are only interested in part of tasks, there is no need of downloading all datasets. The following lines list the datasets needed for different tasks.

- Object detection & instance segmentation: COCO
- SOT: COCO, LaSOT, GOT-10K, TrackingNet
- VOS: DAVIS, Youtube-VOS 2018, COCO, Saliency
- MOT & MOTS (MOT Challenge 17, MOTS Challenge): MOT17, CrowdHuman, ETHZ, CityPerson, COCO, MOTS
- MOT & MOTS (BDD100K): BDD100K


### Object Detection & Instance Segmentation
Please download [COCO](https://cocodataset.org/#home) from the offical website. We use [train2017.zip](http://images.cocodataset.org/zips/train2017.zip), [val2017.zip](http://images.cocodataset.org/zips/val2017.zip) & [annotations_trainval2017.zip](http://images.cocodataset.org/annotations/annotations_trainval2017.zip). We expect that the data is organized as below.
```
${UNICORN_ROOT}
    -- datasets
        -- COCO
            -- annotations
            -- train2017
            -- val2017
```



### SOT
Please download [COCO](https://cocodataset.org/#home), [LaSOT](http://vision.cs.stonybrook.edu/~lasot/download.html), [GOT-10K](http://got-10k.aitestunion.com/downloads) and [TrackingNet](https://tracking-net.org/). Since TrackingNet is very large and hard to download, we only use the first 4 splits (TRAIN_0.zip, TRAIN_1.zip, TRAIN_2.zip, TRAIN_3.zip) rather than the complete 12 splits for the training set. The original TrackingNet zips (put under `datasets`) can be unzipped by the following commands.
```
python3 tools/process_trackingnet.py
```
We expect that the data is organized as below.
```
${UNICORN_ROOT}
    -- datasets
        -- COCO
            -- annotations
            -- train2017
            -- val2017
        -- GOT10K
            -- test
                -- GOT-10k_Test_000001
                -- ...
            -- train
                -- GOT-10k_Train_000001
                -- ...
        -- LaSOT
            -- airplane
            -- basketball
            -- ...
        -- TrackingNet
            -- TEST
            -- TRAIN_0
            -- TRAIN_1
            -- TRAIN_2
            -- TRAIN_3
```

### VOS
Please download [DAVIS](https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip), [Youtube-VOS 2018](https://youtube-vos.org/dataset/), [COCO](https://cocodataset.org/#home), [Saliency](https://drive.google.com/file/d/1qgjvIbeMIBSWfRu6iDrCnY--CbUUhHOb/view?usp=sharing).
The saliency dataset is constructed from [DUTS](http://saliencydetection.net/duts/), [DUT-OMRON](http://saliencydetection.net/dut-omron/), etc.
The downloaded youtube-vos zips can be processed using the following commands.
```
unzip -qq ytbvos18_train.zip
unzip -qq ytbvos18_val.zip
mkdir ytbvos18
mv train ytbvos18/train
mv valid ytbvos18/val
rm -rf ytbvos18_train.zip
rm -rf ytbvos18_val.zip
mv ytbvos18 datasets
```
We expect that the data is organized as below.
```
${UNICORN_ROOT}
    -- datasets
        -- COCO
            -- annotations
            -- train2017
            -- val2017
        -- DAVIS
            -- Annotations
            -- ImageSets
            -- JPEGImages
            -- README.md
            -- SOURCES.md
        -- saliency
            -- image
            -- mask
        -- ytbvos18
            -- train
            -- val
```

### MOT & MOTS (MOT Challenge)
Download [MOT17](https://motchallenge.net/), [CrowdHuman](https://www.crowdhuman.org/), [Cityperson](https://github.com/Zhongdao/Towards-Realtime-MOT/blob/master/DATASET_ZOO.md), [ETHZ](https://github.com/Zhongdao/Towards-Realtime-MOT/blob/master/DATASET_ZOO.md), [MOTS](https://motchallenge.net/) and put them under `datasets` in the following structure:
```
${UNICORN_ROOT}
    -- datasets
        -- Cityscapes
            -- annotations
            -- images
            -- labels_with_ids
        -- COCO
            -- annotations
            -- train2017
            -- val2017
        -- crowdhuman
            -- annotations
            -- CrowdHuman_train
            -- CrowdHuman_val
            -- annotation_train.odgt
            -- annotation_val.odgt
        -- ETHZ
            -- annotations
            -- eth01
            -- eth02
            -- eth03
            -- eth05
            -- eth07
        -- mot
            -- annotations
            -- test
            -- train
        -- MOTS
            -- annotations
            -- test
            -- train
```
unzip CityPersons dataset by 
```
cat Citypersons.z01 Citypersons.z02 Citypersons.z03 Citypersons.zip > c.zip
zip -FF Citypersons.zip --out c.zip
unzip -qq c.zip
```
Unzip CrowdHuman dataset by
```
# unzip the train split
unzip -qq CrowdHuman_train01.zip
unzip -qq CrowdHuman_train02.zip
unzip -qq CrowdHuman_train03.zip
mv Images CrowdHuman_train
# unzip the val split
unzip -qq CrowdHuman_val.zip
mv Images CrowdHuman_val
```

Then, you need to turn the datasets to COCO format:

```shell
python3 tools/convert_mot17_to_coco.py
python3 tools/convert_mot17_to_omni.py --dataset_name mot
python3 tools/convert_crowdhuman_to_coco.py
python3 tools/convert_cityperson_to_coco.py
python3 tools/convert_ethz_to_coco.py
python3 tools/convert_mots_to_coco.py
```



### MOT & MOTS (BDD100K)
We need to download the `detection` set, `tracking` set, `instance seg` set and `tracking & seg` set for training and validation.
For more details about the dataset, please refer to the [offial documentation](https://doc.bdd100k.com/download.html).

We provide the following commands to download and process BDD100K datasets in parallel.
```
cd external/qdtrack
python3 download_bdd100k.py # replace save_dir to your path
bash prepare_bdd100k.sh # replace paths to yours
ln -s <UNICORN_ROOT>/external/qdtrack/data/bdd <UNICORN_ROOT>/datasets/bdd
```
We expect that the data is organized as below
```
${UNICORN_ROOT}
    -- datasets
        -- bdd
            -- images
                -- 10k
                -- 100k
                -- seg_track_20
                -- track
            -- labels
                -- box_track_20
                -- det_20
                -- ins_seg
                -- seg_track_20
```
## Object Tracking Inference

**SOT**

LaSOT
```
python3 tools/test.py unicorn_sot ${exp_name} --dataset lasot --threads 32
python3 tools/analysis_results.py --name ${exp_name}
```

TrackingNet
```
python3 tools/test.py unicorn_sot ${exp_name} --dataset trackingnet --threads 32
python3 external/lib/test/utils/transform_trackingnet.py --tracker_name unicorn_sot --cfg_name ${exp_name}
```


**MOT**

BDD100K
```
cd external/qdtrack
# track
bash tools/dist_test_omni.sh configs/bdd100k/unicorn.py ../../Unicorn_outputs/${exp_name}/latest_ckpt.pth 8 ${exp_name} --eval track
# bbox
python3 tools/eval.py configs/bdd100k/unicorn.py result_omni.pkl --eval bbox
```

MOT Challenge 17
```
python3 tools/track.py -f exps/default/${exp_name} -c <ckpt path> -b 1 -d 1 # using the association strategy in ByteTrack
python3 tools/track_omni.py -f exps/default/${exp_name} -c <ckpt path> -b 1 -d 1 # using the association strategy in QDTrack
python3 tools/interpolation.py # need to change some paths
```

**VOS**

DAVIS-2016
```
python3 tools/test.py unicorn_vos ${exp_name} --dataset dv2016_val --threads 20
cd external/PyDavis16EvalToolbox
python3 eval.py --name_list_path ../../datasets/DAVIS/ImageSets/2016/val.txt --mask_root ../../datasets/DAVIS/Annotations/480p --pred_path ../../test/segmentation_results/unicorn_vos/${exp_name}/ --save_path ../../result.pkl
```

DAVIS-2017
```
python3 tools/test.py unicorn_vos ${exp_name} --dataset dv2017_val --threads 30
cd external/davis2017-evaluation
python3 evaluation_method.py --task semi-supervised --results_path ../../test/segmentation_results/unicorn_vos/${exp_name} --davis_path ../../datasets/DAVIS
```

**MOTS**

MOTSChallenge
```
python3 tools/track_omni.py -f <exp file path> -c <ckpt path> -b 1 -d 1 --mots --mask_thres 0.3
# for train split
cp Unicorn_outputs/${exp_name}/track_results/* ../MOTChallengeEvalKit/res/MOTSres
cd ../MOTChallengeEvalKit
python MOTS/evalMOTS.py
```

BDD100K MOTS
```
cd external/qdtrack
# track
bash tools/dist_test_omni.sh configs/bdd100k_mots/segtrack-frcnn_r50_fpn_12e_bdd10k_fixed_pcan.py ../../Unicorn_outputs/${exp_name}/latest_ckpt.pth 8 ${exp_name} --eval segm --mots
# convert to BDD100K format (bitmask)
python3 tools/to_bdd100k.py configs/bdd100k_mots/segtrack-frcnn_r50_fpn_12e_bdd10k_fixed_pcan.py --res result_omni.pkl --task seg_track --bdd-dir . --nproc 32
# evaluate
bash eval_bdd_submit.sh
```
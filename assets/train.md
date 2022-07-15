# Tutorial for Training

Every experiment is defined using a python file under the `exps/default` folder. Experiments about object detection and object tracking start with `unicorn_det` and `unicorn_track` respectively. In the next paragraphs, ${exp_name} should be replaced with specifc filenames (without .py). For example, if you want to train unicorn with convnext-tiny backbone for object tracking, replace ${exp_name} with unicorn_track_tiny

## Detection & Instance Segmentation

**Single-node Training** 

On a single node with 8 GPUs, run 
```
python3 launch_uni.py --name ${exp_name} --nproc_per_node 8 --batch 64 --mode multiple --fp16 0
```

**Multiple-node Training**

On the master node, run
```
python3 launch_uni.py --name ${exp_name} --nproc_per_node 8 --batch 128 --mode distribute --fp16 0 --nnodes 2 --master_address ${master_address} --node_rank 0
```

On the second node, run
```
python3 launch_uni.py --name ${exp_name} --nproc_per_node 8 --batch 128 --mode distribute --fp16 0 --nnodes 2 --master_address ${master_address} --node_rank 1
```

Testing (Instance Segmentation)
```
python3 tools/eval.py -f exps/default/${exp_name}.py -c Unicorn_outputs/${exp_name}/latest_ckpt.pth -b 64 -d 8 --conf 0.001 --mask_thres 0.3
```

## Unified Tracking (SOT, MOT, VOS, MOTS)

**Single-node Training** 

On a single node with 8 GPUs, run 
```
python3 launch_uni.py --name ${exp_name} --nproc_per_node 8 --batch 16 --mode multiple
```
**Multiple-node Training**

On the master node, run
```
python3 launch_uni.py --name ${exp_name} --nproc_per_node 8 --batch 32 --mode distribute --nnodes 2 --master_address ${master_address} --node_rank 0
```
On the second node, run
```
python3 launch_uni.py --name ${exp_name} --nproc_per_node 8 --batch 32 --mode distribute --nnodes 2 --master_address ${master_address} --node_rank 1
```